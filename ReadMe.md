Experimenting algorithms for background subtraction in dynamic scenes.




# The Process 2 
A full resolution image/frame is viewed as consisting of certain number of blocks. How many blocks we want to divide
a frame into can be decided by the constant BLOCK_SIZE. E.g given an 1920x1080 resolution image, and block size of 60x60
we will get 1920/60 = 32 horizontal blocks and 18 vertical block for a total of 576 blocks per frame. Each such block
represents a specific position of the frame. Block 1 is the 60x60 block of pixels located in the top left corner
block 1 is the 60x60 block of pixels located just to the right of block 1 etc. And block 576 represent the 60x60 
block of pixels located in the bottom right corner.  
A stream of video will consist of a given number of frames. Each individual frame will thus contribute 576 blocks.

We create a vector of vectors to store all blocks from all frames: vector<vector<Mat>> blocks(NR_H_BLOCKS * NR_V_BLOCKS);
The outer "blocks" vector is of size 576 and at each index j it contains another vector responsable for storing
positional block j from all frames. 

A Patch is class 