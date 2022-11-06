Experimenting algorithms for background subtraction in dynamic scenes.

In this readme we add some explanation of the logic/process and thinking behind the code. So we can
more easily understand if we ever need to review it. 

# How to run the program
1. Run make from whithin build folder.
2. Then run ./Main ../data/video_frames patchpca -V=.75 -N=100
 First arg is path to file, second is the string of the func to run,
 -V variance to retain (this can be used differentyl by passing an int value and treating the variable 
 as max nr of components instead.). 
 -N is the nr of frames to use from file path source supplied.
-K is NR_OF_COMP_TO_USE to use that we can specify as command line option. 
but I have commented it out in main because I specifying this in the Constants.hpp instead.

# The Process 
A full resolution image/frame is viewed as consisting of certain number of blocks. 
How many blocks we want to dividea frame into can be decided by the constant BLOCK_SIZE. 
E.g given an 1920x1080 resolution image, and block size of 60x60 we will get 1920/60 = 32 
horizontal blocks and 18 vertical block for a total of 576 blocks per frame. Each such block
represents a specific position of the frame. Block 0 is the 60x60 block of pixels located 
in the top left corner block 1 is the 60x60 block of pixels located just to the right of 
block 0 etc. And block 575 represent the 60x60 block of pixels located in the bottom right 
corner. A stream of video will consist of a given number of frames. Each individual frame 
will thus contribute 576 blocks.

In Main.cpp we create a vector "called blocks" to store other vectors. We will need as many
of these "inner" vector as the number of blocks we have decided to divide a frame into.
The declarations is : vector<vector<Mat>> blocks(NR_H_BLOCKS * NR_V_BLOCKS);
Each inner vector stores the "cutout" / "cropped" part of the original frame for all
frames in video sequence.

Now that we have our stream of frames divided/cropped into parts/blocks where each is stored
separately we can proceed with PCA for each part/block.

The Patch class is used to do pca on all frames for a given block and store the principal 
components/eigenvectors for that block. A patch is then used for projecting/backprojecting 
from the subspace. We will of course have as many patches as we have blocks. 

After we a patch containing the eigevecs/princip components for a given block
we then a patch and its corresponding smalled sized frames and project each such
on the lower dimensional subspace so that we get coords the lower-dimensional subspace.
This means now that each frame previosly stored in Mat structure now becomes just 
column vector. So after all the frames belonging to a give block have been projected
to the subspace, they can be saved in a new Mat structure. But this structure
now represents "all the frames in the video seqs belonging to a certain block". 

So by using a "vector<Mat> blocksAsCoords" we can store data from all frames for all blocks


