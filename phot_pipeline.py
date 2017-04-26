import photutils
from ccdproc import CCDData, Combiner
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from matplotlib import patches
import glob
from photutils import CircularAperture, CircularAnnulus
from photutils import centroid_2dg, aperture_photometry
import numpy as np
from astropy.time import Time
import pdb
import es_gen
from copy import deepcopy
import yaml
import os

class phot:
    def __init__(self,paramFile='parameters/phot_parameters.yaml'):
        """ Photometry class
    
        Parameters
        ------
        paramFile: str
            Location of the YAML file that contains the photometry parameters
        Properties
        -------
        paramFile: str
            Same as paramFile above
        param: dict
            The photometry parameters like file names, aperture sizes, guess locations
        fileL: list
            The files on which photometry will be performed
        nImg: int
            Number of images in the sequence
        """
        self.paramFile = paramFile
        self.param = yaml.load(open(paramFile))
        self.fileL = np.sort(glob.glob(self.param['procFiles']))
        self.nImg = len(self.fileL)
        xCoors, yCoors = [], []
        positions = self.param['refStarPos']
        self.nsrc = len(positions)
        self.srcApertures = CircularAperture(positions,r=self.param['apRadius'])
        self.xCoors = self.srcApertures.positions[:,0]
        self.yCoors = self.srcApertures.positions[:,1]
        self.bkgApertures = CircularAnnulus(positions,r_in=self.param['backStart'],r_out=self.param['backEnd'])
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
        self.srcNames[0] = 'src'
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        self.photFile = 'tser_data/phot/phot_'+self.dataFileDescrip+'.fits'
        self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'

    def showStarChoices(self):
        """ Show the star choices for photometry """
        fig, ax = plt.subplots()
        
        img, head = getImg(self.fileL[self.nImg/2])
        #t = Time(head['DATE-OBS']+'T'+head['TIME-OBS'])
        
        imData = ax.imshow(img,cmap='viridis',vmin=0,vmax=1.0e4,interpolation='nearest')
        ax.invert_yaxis()
        rad, txtOffset = 50, 20
        ax.scatter(self.xCoors, self.yCoors, s=rad, facecolors='none', edgecolors='r')
        for ind, onePos in enumerate(self.srcApertures.positions):
            
            #circ = plt.Circle((onePos[0], onePos[1]), rad, color='r')
            #ax.add_patch(circ)
            if ind == 0:
                name='src'
            else:
                name=str(ind)
            ax.text(onePos[0]+txtOffset,onePos[1]+txtOffset,name,color='white')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        fig.colorbar(imData,label='Counts')
        fig.show()
        fig.savefig('plots/photometry/star_labels/st_labels.pdf')

    def showStamps(self,img=None,custPos=None):
        """Shows the fixed apertures on the image with postage stamps surrounding sources """ 
        
        ##  Calculate approximately square numbers of X & Y positions in the grid
        numGridY = int(np.floor(np.sqrt(self.nsrc)))
        numGridX = int(np.ceil(float(self.nsrc) / float(numGridY)))
        fig, axArr = plt.subplots(numGridY, numGridX)
        
        ## Get the data
        if img is None:
            img, head = getImg(self.fileL[self.nImg/2])
        
        boxsize = self.param['boxFindSize']
        
        if custPos is None:
            showApPos = self.srcApertures.positions
        else:
            showApPos = custPos
        
        for ind, onePos in enumerate(showApPos):
            ax = axArr.ravel()[ind]
            
            yStamp = np.array(onePos[1] + np.array([-1,1]) * boxsize,dtype=np.int)
            xStamp = np.array(onePos[0] + np.array([-1,1]) * boxsize,dtype=np.int)
            
            stamp = img[yStamp[0]:yStamp[1],xStamp[0]:xStamp[1]]
            
            imData = ax.imshow(stamp,cmap='viridis',vmin=0,vmax=1.2e4,interpolation='nearest')
            ax.set_title(self.srcNames[ind])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            circ = plt.Circle((onePos[0] - xStamp[0],onePos[1] - yStamp[0]),
                              self.param['apRadius'],edgecolor='red',facecolor='none')
            ax.add_patch(circ)
        
        fig.colorbar(imData,label='Counts')
        
        totStamps = numGridY * numGridX
        
        for ind in np.arange(self.nsrc,totStamps):
            ## Make any extra postage stamps blank
            ax = axArr.ravel()[ind]
            
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #self.srcApertures.plot(indices=ind,color='red')
            #ax.set_xlim(onePos[0] - boxsize,onePos[0] + boxsize)
            #ax.set_ylim(onePos[1] - boxsize,onePos[1] + boxsize)
        
            
        fig.show()
        fig.savefig('plots/photometry/postage_stamps/postage_stamps.pdf')
        
    
    def get_allimg_cen(self,recenter=False):
        
        if os.path.exists(self.centroidFile) and (recenter == False):
            cenArr, head = getImg(self.centroidFile)
        else:
            ndim=2
            cenArr = np.zeros((self.nImg,self.nsrc,ndim))
            #for ind, oneFile in enumerate(self.fileL):
            for ind, oneFile in enumerate(self.fileL):
                img, head = getImg(oneFile)
                allX, allY = self.get_allcen_img(img)
                cenArr[ind,:,0] = allX
                cenArr[ind,:,1] = allY
            hdu = fits.PrimaryHDU(cenArr)
            hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with centroids')
            hdu.header['NIMG'] = (self.nImg,'Number of images')
            hdu.header['AXIS1'] = ('dimension','dimension axis X=0,Y=1')
            hdu.header['AXIS2'] = ('src','source axis')
            hdu.header['AXIS3'] = ('image','image axis')
            hdu.header['BOXSZ'] = (self.param['boxFindSize'],'half-width of the box used for source centroids')
            
            self.add_filenames_to_header(hdu)
            HDUList = fits.HDUList([hdu])
            HDUList.writeto(self.centroidFile,overwrite=True)
            head = hdu.header
            
        self.cenArr = cenArr
        self.cenHead = head
    
    def get_allcen_img(self,img,showStamp=False):
        """ Gets the centroids for all sources in one image """
        allX, allY = [], []
        for ind, onePos in enumerate(self.srcApertures.positions):
            xcen, ycen = self.get_centroid(img,onePos[0],onePos[1])
            allX.append(xcen)
            allY.append(ycen)
        
        if showStamp == True:
            posArr = np.vstack((allX,allY)).transpose()
            self.showStamps(img=img,custPos=posArr)
        return allX, allY
    
    def get_centroid(self,img,xGuess,yGuess):
        """ Get the centroid of a source given an x and y guess 
        Takes the self.param['boxFindSize'] to define the search box
        """
        boxSize=self.param['boxFindSize']
        shape = img.shape
        minX = int(np.max([xGuess - boxSize,0.]))
        maxX = int(np.min([xGuess + boxSize,shape[1]-1]))
        minY = int(np.max([yGuess - boxSize,0.]))
        maxY = int(np.min([yGuess + boxSize,shape[1]-1]))
        subimg = img[minY:maxY,minX:maxX]
        
        xcenSub,ycenSub = centroid_2dg(subimg)
        xcen = xcenSub + minX
        ycen = ycenSub + minY
        
        return xcen, ycen
    
    def add_filenames_to_header(self,hdu):
        """ Uses fits header cards to list the files"""
        for ind, oneFile in enumerate(self.fileL):
            hdu.header['FIL'+str(ind)] = (os.path.basename(oneFile),'file name')
        
    
    def do_phot(self):
        """ Does photometry using the centroids found in get_allimg_cen 
        """
        self.get_allimg_cen()
        
        photArr = np.zeros((self.nImg,self.nsrc))
        
        jdArr = []
        
        for ind,oneImg in enumerate(self.fileL):
            if np.mod(ind,15) == 0:
                print("On "+str(ind)+' of '+str(len(self.fileL)))
            
            img, head = getImg(oneImg)
            t = Time(head['DATE-OBS']+'T'+head['TIME-OBS'])
            jdArr.append(t.jd)
            
            self.srcApertures.positions = self.cenArr[ind]
            self.bkgApertures.positions = self.cenArr[ind]  
            
            rawPhot = aperture_photometry(img,self.srcApertures)
            bkgPhot = aperture_photometry(img,self.bkgApertures)
            bkgVals = bkgPhot['aperture_sum'] / self.bkgApertures.area() * self.srcApertures.area()

            ## Background subtracted fluxes
            srcPhot = rawPhot['aperture_sum'] - bkgVals
            photArr[ind,:] = srcPhot
        
        ## Save the photometry results
        hdu = fits.PrimaryHDU(photArr)
        hdu.header['NSOURCE'] = (self.nsrc,'Number of sources with photometry')
        hdu.header['NIMG'] = (self.nImg,'Number of images')
        hdu.header['AXIS1'] = ('src','source axis')
        hdu.header['AXIS2'] = ('image','image axis')
        hdu.header['SRCNAME'] = (self.param['srcName'], 'Source name')
        hdu.header['NIGHT'] = (self.param['nightName'], 'Night Name')
        hdu.header['APRADIUS'] = (self.param['apRadius'], 'Aperture radius (px)')
        hdu.header['BKGSTART'] = (self.param['backStart'], 'Background Annulus start (px)')
        hdu.header['BKGEND'] = (self.param['backEnd'], 'Background Annulus end (px)')
        hdu.header['BOXSZ'] = (self.cenHead['BOXSZ'], 'half-width of the box used for source centroiding')
        hdu.header['JDREF'] = (self.param['jdRef'], ' JD reference offset to subtract for plots')
        
        self.add_filenames_to_header(hdu)
        
        hduTime = fits.ImageHDU(jdArr)
        hduTime.header['UNITS'] = ('days','JD time, UT')
        
        hduCen = fits.ImageHDU(data=self.cenArr,header=self.cenHead)
        
        hdu.name = 'Photometry'
        hduTime.name = 'Time'
        hduCen.name = 'Centroids'
        HDUList = fits.HDUList([hdu,hduTime,hduCen])
        
        HDUList.writeto(self.photFile,overwrite=True)
        
    def plot_phot(self,offset=0.,refCorrect=False):
        """ Plots previously calculated photometry """
        photArr, head = getImg(self.photFile)
        jdArr, timeHead = getImg(self.photFile,ext=1)
        
        jdRef = self.param['jdRef']
        
        fig, ax = plt.subplots()
        
        if refCorrect == True:
            yCorrected = self.refSeries(photArr)
            ax.plot(jdArr - jdRef,yCorrected,label='data')
        else:
            for oneSrc in range(self.nsrc):
                yFlux = photArr[:,oneSrc]
                yNorm = yFlux / np.median(yFlux)
                if oneSrc == 0:
                    pLabel = 'Src'
                else:
                    pLabel = 'Ref '+str(oneSrc)
                yplot = yNorm - offset * oneSrc
                ## To avoid repeat colors, switch to dashed lins
                if oneSrc > 10: linestyle='dashed'
                else: linestyle= 'solid'
                ax.plot(jdArr - jdRef,yplot,label=pLabel,linestyle=linestyle)
        
            ax.set_title('Src Ap='+str(head['APRADIUS'])+',Back=['+str(head['BKGSTART'])+','+
                         str(head['BKGEND'])+']')
        ax.set_xlabel('JD - '+str(jdRef))
        ax.set_ylabel('Normalized Flux + Offset')
        #ax.set_ylim(0.94,1.06)
        ax.legend(loc='best',fontsize=10)
        fig.show()
        if refCorrect == True:
            fig.savefig('plots/photometry/tser_refcor/refcor_01.pdf')
    
    def refSeries(self,photArr,reNorm=False,custSrc=None):
        """ Average together the reference stars
        Parameters
        -------------
        reNorm: bool
            Re-normalize all stars before averaging? If set all stars have equal weight
            Otherwise, the stars are summed together, which weights by flux
        custSrc: arr
            Custom sources to use in the averaging (to include/exclude specific sources)
        """
        combRef = []
        if custSrc == None:
            refArray = np.arange(1,self.nsrc)
        else:
            refArray = custSrc
        
        for oneSrc in refArray:
            if reNorm == True:
                refSeries = photArr[:,oneSrc]
                refSeries = refSeries / np.median(refSeries)
            else:
                refSeries = photArr[:,oneSrc]
            
            if oneSrc == 1:
                combRef = refSeries
            else:
                combRef = combRef + refSeries
        yCorrected = photArr[:,0] / combRef
        yCorrNorm = yCorrected / np.median(yCorrected)
        return yCorrNorm

def getImg(path,ext=0):
    """ Load an image from a given path and extensions"""
    HDUList = fits.open(path)
    img = HDUList[ext].data
    head = HDUList[ext].header
    HDUList.close()
    return img, head

# # ## Compare to SLC curve
#
# # In[63]:
#
# lc = ascii.read('slc_curve/k1255.hjd',data_start=0,names=['HJD'])
#
# quickfl = ascii.read('slc_curve/k1255.dR7.r12345689101112',data_start=0,names=['Fl'])
#
# lc.add_column(quickfl['Fl'])
#
# x = lc['HJD'] - 1.
# y = 10**(0.4 * lc['Fl'])
#
#
# # In[ ]:
#
# yfit = es_gen.yCorrNorm3
#
#
# # In[72]:
#
# fig, ax = plt.subplots(figsize=(12,12))
# #ax.plot(jdArr - jdRef,yCorrNorm3,'o',label='Choice Ref Norm Sum')
# ax.plot(jdArr - jdRef,yCorrNorm3,label='Choice Ref Norm Sum')
# #ax.plot(x,y,'o',label='slc pipelin')
# ax.plot(x,y,label='slc pipelin')
# ax.legend(loc='lower right')
# ax.set_xlabel('JD - '+str(jdRef))
# ax.set_ylabel('Normalized Flux')
# fig.savefig('reference_method.pdf')


# In[ ]:



