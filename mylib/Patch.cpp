#include "../include/Patch.hpp"
#include <opencv2/core.hpp>
#include <iostream>


// Def class static consts 
const int Patch::PNR_ROWS{60},Patch::PNR_COLS{60}, Patch::PNR_PIXELS{3600} ;

//Constructor
Patch::Patch(int id):ID(id){

}

// Patch::~Patch(){

// }
int Patch::getID(){
    return ID;
}

