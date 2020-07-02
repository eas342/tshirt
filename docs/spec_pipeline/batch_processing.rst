
Batch Processing
----------------
A batch object can iterate over any spec object.
The command below shows how to create a :class:`tshirt.pipeline.spec_pipeline.batch_spec` object to iterate over many different :class:`tshirt.pipeline.spec_pipeline.spec` methods.
Here is an example on how to run :any:`spec_pipeline.spec.plot_wavebin_series` over the batch object.

.. code-block:: python

   bspec = spec_pipeline.batch_spec(batchFile='corot1_batch_file.yaml')
   bspec.batch_run('plot_wavebin_series', nbins=1, interactive=True) 
   
This could be done on any method that spec can.
One way to check if the batch file is working and test a method is to have the batch object spit out a :class:`tshirt.pipeline.spec_pipeline.spec` object using :any:`spec_pipeline.batch_spec.return_spec_obj`.

.. code-block:: python

   bspec = spec_pipeline.batch_spec(batchFile='corot1_batch_file.yaml')
   spec = bspec.return_spec_obj()
   spec.plot_one_spec()

