There are multiple steps to run this code.  Think of it as a pipeline, since
there are many operations that take a considerable amount of time.

   preprocess xml data  -->  extract sub images  -->  train  -->  use classifier

Each of these steps is isolated and will be run separately.


== Preprocessing of XML data ==

This is done using the preprocess.py script.  Call it with --help for usage
information.  It takes a test directory and a training directory (if you have
split your data into a test directory and a training directory), and outputs a
tab-separated file (tsv) for training and testing containing the relevant
labels and fields extracted from the XML files.

Example:

  $ python preprocess.py --train data/training --test data/test --outdir labels

This command will create

  labels/
  |-- test.tsv
  `-- train.tsv


== Overlay Boxes (optional) ==

There is an optional script that can be used to visualize the locations of
these bounding boxes.  It is the overlayboxesonimages.py script.  You can call
the --help option for more information.

Example:

  $ python overlayboxesonimages.py --outdir red-boxes labels/train.tsv

This will take the images referenced in the tsv file, and draw red bounding
boxes for all of the images and place those copies into the red-boxes
directory.


== Extraction ==

One of the XML files contains bounding boxes for the images.  These bounding
boxes will be specified in the tsv files generated in the previous step.  This
step uses the tsv file and creates individual images for each of the bounding
boxes.  It will also make these sub-images the same size.  This script is the
create_image_cache.py script.  Use the --help option for more information.

Example:

  $ python create_image_cache.py labels/test.tsv subimages/test
  $ python create_image_cache.py labels/train.tsv subimages/train

You will now see a single image for each of the labeled data in the specified
directories.  There will also be a cache file created in each of the
directories that contains the pickled contents of the images with the labels,
for use in training and testing.


== Training ==

Training is now performed using the labels from the training tsv file and the
image cache.  This is done using train.py

Example:

  $ python train.py \
      subimages/train/cache.pkl \
      subimages/test/cache.pkl   \
      --output classifier.dat

This example will take the two tsv files as well as the two image caches
created in previous steps.  It also specifies to output the classifier to the
file classifier.dat, which can be used to classify in later steps.


== Classifying ==

Classifying requires only a subimage in the same format as the ones generated
in the Extraction phase.  They should be 68x40 pixels.  Place the images you
want classified into a directory and then call the classify.py script.

Example:

  $ python classify.py \
      --input toclassify/             \
      --output classified-labels.tsv  \
      --classifier classifier.dat

This command will take images from toclassify and will load the previously
generated classifier.dat from the Training phase, and will output a set of
labels for the images.  You are free to look inside of this script to see how
you can use it to classify inside of your python script.

