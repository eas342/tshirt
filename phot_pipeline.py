import photutils
from ccdproc import CCDData, Combiner
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
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
        if 'srcGeometry' not in self.param:
            self.geometry = 'Circular'
        if self.geomety == 'Circular':
            self.srcApertures = CircularAperture(positions,r=self.param['apRadius'])
        self.xCoors = self.srcApertures.positions[:,0]
        self.yCoors = self.srcApertures.positions[:,1]
        self.bkgApertures = CircularAnnulus(positions,r_in=self.param['backStart'],r_out=self.param['backEnd'])
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
        self.srcNames[0] = 'src'
        self.dataFileDescrip = self.param['srcNameShort'] + '_'+ self.param['nightName']
        self.photFile = 'tser_data/phot/phot_'+self.dataFileDescrip+'.fits'
        self.centroidFile = 'centroids/cen_'+self.dataFileDescrip+'.fits'
        
        
        
    def get_default_im(self,img=None,head=None):
        """ Get the default image for postage stamps or star identification maps"""
        ## Get the data
        if img is None:
            img, head = getImg(self.fileL[self.nImg/2])
        
        return img, head
    
    def get_default_cen(self,custPos=None):
        """ Get the default centroids for postage stamps or star identification maps"""
        if custPos is None:
            showApPos = self.srcApertures.positions
        else:
            showApPos = custPos
        
        return showApPos
    
    def showStarChoices(self,img=None,head=None,custPos=None):
        """ Show the star choices for photometry """
        fig, ax = plt.subplots()
        
        img, head = self.get_default_im(img=img,head=None)
        
        imData = ax.imshow(img,cmap='viridis',vmin=0,vmax=1.0e4,interpolation='nearest')
        ax.invert_yaxis()
        rad, txtOffset = 50, 20
        ax.scatter(self.xCoors, self.yCoors, s=rad, facecolors='none', edgecolors='r')
        
        showApPos = self.get_default_cen(custPos=custPos)
        
        for ind, onePos in enumerate(showApPos):
            
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

    def showStamps(self,img=None,head=None,custPos=None):
        """Shows the fixed apertures on the image with postage stamps surrounding sources """ 
        
        ##  Calculate approximately square numbers of X & Y positions in the grid
        numGridY = int(np.floor(np.sqrt(self.nsrc)))
        numGridX = int(np.ceil(float(self.nsrc) / float(numGridY)))
        fig, axArr = plt.subplots(numGridY, numGridX)
        
        img, head = self.get_default_im(img=img,head=head)
        
        boxsize = self.param['boxFindSize']
        
        showApPos = self.get_default_cen(custPos=custPos)
        
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
        fig.savefig('plots/photometry/postage_stamps/stamps_'+self.dataFileDescrip+'.pdf')
        
    def showCustSet(self,index=None,ptype='Stamps',defaultCen=False):
        """ Show a custom stamp or star identification plot for a given image index 
        
        Parameters
        --------------
        index: int
            Index of the image/centroid to show
        ptype: str
            Plot type - 'Stamps' for postage stamps
                        'Map' for star identification map
        defaultCen: bool
            Use the default centroid? If true, it will use the guess centroids
        """
        self.get_allimg_cen()
        if index == None:
            index = self.nImg / 2
        
        img, head = getImg(self.fileL[index])
        
        if defaultCen == True:
            cen = self.srcApertures.positions
        else:
            cen = self.cenArr[index]
        
        if ptype == 'Stamps':
            self.showStamps(custPos=cen,img=img,head=head)
        elif ptype == 'Map':
            self.showStarChoices(custPos=cen,img=img,head=head)
        else:
            print('Unrecognized plot type')
            
    
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
        errArr = np.zeros_like(photArr)
        
        jdArr = []
        
        for ind,oneImg in enumerate(self.fileL):
            if np.mod(ind,15) == 0:
                print("On "+str(ind)+' of '+str(len(self.fileL)))
            
            img, head = getImg(oneImg)
            t = Time(head['DATE-OBS']+'T'+head['TIME-OBS'])
            jdArr.append(t.jd)
            
            self.srcApertures.positions = self.cenArr[ind]
            self.bkgApertures.positions = self.cenArr[ind]  
            
            if 'RDNOISE1' in head:
                readNoise = float(head['RDNOISE1'])
            else:
                readNoise = 0.
                print('Warning, no read noise specified')
            
            err = np.sqrt(img + readNoise**2) ## Should already be gain-corrected
            
            rawPhot = aperture_photometry(img,self.srcApertures,error=err)
            bkgPhot = aperture_photometry(img,self.bkgApertures,error=err)
            bkgVals = bkgPhot['aperture_sum'] / self.bkgApertures.area() * self.srcApertures.area()
            bkgValsErr = bkgPhot['aperture_sum_err'] / self.bkgApertures.area() * self.srcApertures.area()
            
            ## Background subtracted fluxes
            srcPhot = rawPhot['aperture_sum'] - bkgVals
            srcPhotErr = np.sqrt(rawPhot['aperture_sum_err']**2 + bkgValsErr**2)
            photArr[ind,:] = srcPhot
            errArr[ind,:] = srcPhotErr
        
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
        
        hduErr = fits.ImageHDU(data=errArr)
        hduErr.name = 'Phot Err'
        hduCen = fits.ImageHDU(data=self.cenArr,header=self.cenHead)
        
        hdu.name = 'Photometry'
        hduTime.name = 'Time'
        hduCen.name = 'Centroids'
        HDUList = fits.HDUList([hdu,hduErr,hduTime,hduCen])
        
        HDUList.writeto(self.photFile,overwrite=True)
    
    def plot_phot(self,offset=0.,refCorrect=False,ax=None,fig=None,showLegend=True):
        """ Plots previously calculated photometry """
        HDUList = fits.open(self.photFile)
        photHDU = HDUList['PHOTOMETRY']
        photArr = photHDU.data
        head = photHDU.header
        
        jdHDU = HDUList['TIME']
        jdArr = jdHDU.data
        timeHead = jdHDU.header
        
        jdRef = self.param['jdRef']
        
        if ax == None:
            fig, ax = plt.subplots()
        
        if refCorrect == True:
            yCorrected = self.refSeries(photArr)
            ax.plot(jdArr - jdRef,yCorrected,label='data',marker='o',linestyle='',markersize=3.)
        
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
                if oneSrc >= 10: linestyle='dashed'
                else: linestyle= 'solid'
                ax.plot(jdArr - jdRef,yplot,label=pLabel,linestyle=linestyle)
        
            ax.set_title('Src Ap='+str(head['APRADIUS'])+',Back=['+str(head['BKGSTART'])+','+
                         str(head['BKGEND'])+']')
        ax.set_xlabel('JD - '+str(jdRef))
        ax.set_ylabel('Normalized Flux + Offset')
        #ax.set_ylim(0.94,1.06)
        if showLegend == True:
            ax.legend(loc='best',fontsize=10)
        fig.show()
        if refCorrect == True:
            fig.savefig('plots/photometry/tser_refcor/refcor_01.pdf')
    
    def refSeries(self,photArr,reNorm=False,custSrc=None,sigRej=5.):
        """ Average together the reference stars
        Parameters
        -------------
        reNorm: bool
            Re-normalize all stars before averaging? If set all stars have equal weight
            Otherwise, the stars are summed together, which weights by flux
        custSrc: arr
            Custom sources to use in the averaging (to include/exclude specific sources)
        sigRej: int
            Sigma rejection threshold
        """
        combRef = []
        
        srcArray = np.arange(self.nsrc,dtype=np.int)
        
        if custSrc == None:
            refArrayTruth = (srcArray == 0)
        else:
            refArrayTruth = np.ones(self.nrc,dtype=np.bool)
            for ind, oneSrc in enumerate(custSrc):
                if oneSrc in srcArray:
                    refArrayTruth[ind] = False
        
        refMask = np.tile(refArrayTruth,(self.nImg,1))
        refPhot = np.ma.array(photArr,mask=refMask)
        
        ## Normalize all time series
        norm1D = np.nanmedian(photArr,axis=0)
        norm2D = np.tile(norm1D,(self.nImg,1))
        
        normPhot = refPhot / norm2D
        
        ## Find outliers
        # Median time series
        medTimSeries1D = np.nanmedian(normPhot,axis=1)
        medTimSeries2D = np.tile(medTimSeries1D,(self.nsrc,1)).transpose()
        
        # Absolute deviation
        absDeviation = np.abs(normPhot - medTimSeries2D)
        # Median abs deviation of all reference photometry
        MADphot = np.nanmedian(absDeviation)
        # Points that deviate above threshold
        badP = (absDeviation > sigRej * np.ones((self.nImg,self.nsrc),dtype=np.float) * MADphot)
        
        normPhot.mask = refMask | badP
        refPhot.mask = refMask | badP
        
        if reNorm == True:
            ## Weight all stars equally
            combRef = np.nanmean(normPhot,axis=1)
        else:
            ## Weight by the flux, but only for the valid points left
            weights = np.ma.array(norm2D,mask=normPhot.mask)
            ## Make sure weights sum to 1.0 for each time point (since some sources are missing)
            weightSums1D = np.nansum(weights,axis=1)
            weightSums2D = np.tile(weightSums1D,(self.nsrc,1)).transpose()
            weights = weights / weightSums2D
            combRef = np.nansum(normPhot * weights,axis=1)
        
        yCorrected = photArr[:,0] / combRef
        yCorrNorm = yCorrected / np.nanmedian(yCorrected)
        return yCorrNorm


class prevPhot(phot):
    """ Loads in previous photometry from FITS data. Inherits functions from the phot class
    
    Parameters:
    ---------------
    photFile: str
        Directory of the photometry file
    """
    def __init__(self,photFile='tser_data/phot/phot_kic1255_UT2016_06_12.fits'):
        self.photFile = photFile
        HDUList = fits.open(self.photFile)
        photHead = HDUList[0].header
        
        self.fileL = 'origProcFiles'
        
        self.nImg = photHead['NIMG']
        
        #xCoors, yCoors = [], []
        #positions = self.param['refStarPos']
        self.nsrc = photHead['NSOURCE']
        
        #self.srcApertures = CircularAperture(positions,r=self.param['apRadius'])
        #self.xCoors = self.srcApertures.positions[:,0]
        #self.yCoors = self.srcApertures.positions[:,1]
        #self.bkgApertures = CircularAnnulus(positions,r_in=self.param['backStart'],r_out=self.param['backEnd'])
        self.srcNames = np.array(np.arange(self.nsrc),dtype=np.str)
        self.srcNames[0] = 'src'
        
        self.dataFileDescrip = os.path.splitext(os.path.basename(self.photFile))
        self.param = {}
        self.param['srcName'] = photHead['SRCNAME']
        self.param['nightName'] = photHead['NIGHT']
        self.param['apRadius'] = photHead['APRADIUS']
        self.param['backStart'] = photHead['BKGSTART']
        self.param['backEnd'] = photHead['BKGEND']
        self.param['jdRef'] = photHead['JDREF']
        self.param['boxFindSize'] = photHead['BOXSZ']
        self.centroidFile = self.photFile
        
        HDUList.close()

def getImg(path,ext=0):
    """ Load an image from a given path and extensions"""
    HDUList = fits.open(path)
    img = HDUList[ext].data
    head = HDUList[ext].header
    HDUList.close()
    return img, head

def allTser(refCorrect=False):
    """ Plot all time series for KIC 1255 """
    
    allFits = glob.glob('tser_data/phot/phot_kic1255_UT????_??_??.fits')
    epochs = [2457551.822808,  2457553.7834694,  2457581.8862828,
              2457583.8469442,  2457585.8076056]
    
    nNights = len(allFits)
    
    fig = plt.figure(figsize=(8,6))
    nY, nX = (3,2)
    gs = gridspec.GridSpec(nY,nX,wspace=0.0,hspace=0.7)
    yArr, xArr = np.mgrid[0:nY,0:nX]
    yRavel = yArr.ravel()
    xRavel = xArr.ravel()
    
    kepHDUList = fits.open('tser_data/reference_dat/avg_bin_kep.fits')
    kepLC = kepHDUList[1].data
    tKepler = kepLC['BINMID'] * kepHDUList[1].header['PERIOD'] ## time in days
    fKepler = kepLC['YBIN']
    
    for ind, oneFits in enumerate(allFits):
        phot = prevPhot(photFile=oneFits)
        thisAx = plt.subplot(gs[ind])        
        phot.plot_phot(ax=thisAx,fig=fig,showLegend=False,refCorrect=refCorrect)
        if refCorrect == True:
            thisAx.set_ylim(0.985,1.015)
        else: thisAx.set_ylim(0.9,1.1)
        thisAx.set_xlim(0.65,0.99)
        
        thisAx.set_title(phot.param['nightName'].replace('_','-'))
        ## Hide the y labels for stars not on the left. Also shorten title
        if xRavel[ind] == 0:
            thisAx.set_ylabel('Norm Fl')
        else:
            thisAx.yaxis.set_visible(False)
        ## Show a legend with the last one
        if ind == len(allFits) - 1:
            thisAx.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=2)
        
        ## Show transits for reference corrected photometry
        if refCorrect == True:
            thisAx.axvline(x=epochs[ind] - phot.param['jdRef'],linewidth=2,color='red',alpha=0.5)
            ## Over-plot Kepler Avg SC Light curves
            thisAx.plot(tKepler + epochs[ind] - phot.param['jdRef'],fKepler,color='blue',linewidth=2,alpha=0.5)
    
    fig.show()
    if refCorrect == True:
        fig.savefig('plots/photometry/tser_refcor/all_kic1255.pdf')
    else:
        fig.savefig('plots/photometry/tser_allstar/all_kic1255.pdf')

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



