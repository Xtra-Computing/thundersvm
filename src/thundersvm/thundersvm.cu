//
// Created by jiashuai on 17-9-14.
//


#include <thundersvm/util/log.h>
#include <thundersvm/kernel/testkernel.h>

INITIALIZE_EASYLOGGINGPP
int main(){
    LOG(INFO)<<"kernel start";
    test<<<1,1>>>();
    LOG(INFO)<<"kernel end";
    return 0;
}
