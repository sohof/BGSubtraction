#ifndef MY_CONSTS
#define MY_CONSTS


namespace myConsts{

inline constexpr int NR_CHANNELS{3};
inline constexpr int COLOR_CODE{1}; // 0 is grayscale, 1 is bgr 3-channel color.
inline constexpr int BLOCK_SIZE{60};
inline constexpr int PNR_ROWS{60}; // P stands for Patch.  PNR = nr rows in a patch.
inline constexpr int PNR_COLS{60};
inline constexpr int PNR_PIXELS{360};
}
#endif