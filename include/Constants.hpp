#ifndef MY_CONSTS
#define MY_CONSTS
/*
 ATT: Working with Mat structure. This structure can be somewhat confusing. If we construct a 
 Mat A(2,3). It means create 2 rows and 3 columns. But this parameter order shifts depending 
 on what function you use. For ex, doing A.size() gives a size (width x height) which corresponds to (nrColumns x nrRows).
 So for our A that would print size= (3 x 2). So it seems in every instance we have to be mindful of what order the params
 are expected and given. 
*/

// String names for the different matrices used. OUTSIDE OF THE myConsts namespace!!
const std::string Wstr = "W";
const std::string bstr = "b";
const std::string dA = "dA"; 
const std::string dW = "dW"; 
const std::string db = "db"; 
// Having different names for some constants is more intuitive when reading/writing the code.
namespace myConsts {
inline constexpr char DEFAULT_WIN_NAME[] = "Image"; // default window name to use.
inline constexpr bool DEBUG = 0;
inline constexpr int COLOR_CODE{1}; // 0 is grayscale, 1 is bgr 3-channel color.
inline constexpr int NR_CHANNELS{3};
inline constexpr int BLOCK_SIZE{300}; 

// P stands for Patch.  PNR = nr rows in a patch. 
inline constexpr int PNR_ROWS{BLOCK_SIZE}, PNR_COLS{BLOCK_SIZE}, PNR_PIXELS{BLOCK_SIZE*BLOCK_SIZE};
inline constexpr int PNR_MAX_COMPONENTS{8};

inline constexpr int hidden_layers[]={5,2};// the input & last layer size are calc./set in layer_sizes func.

}
#endif