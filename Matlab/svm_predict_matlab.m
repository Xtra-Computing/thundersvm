function svm_train_matlab(a)
    str = {'thundersvm-predict'}
	str2 = [str a]
    if ispc
        if not(libisloaded('thundersvm'))
            loadlibrary('../build/bin/Debug/thundersvm.dll', '../include/thundersvm/svm_interface_api.h')
        end
        calllib('thundersvm', 'thundersvm_predict', length(str2), str2)
    elseif ismac
        if  not(libisloaded('libthundersvm'))
            loadlibrary('../build/lib/libthundersvm.dylib', '../include/thundersvm/svm_interface_api.h')
        end
        calllib('libthundersvm', 'thundersvm_predict', length(str2), str2)
    elseif isunix
        if  not(libisloaded('libthundersvm'))
            loadlibrary('../build/lib/libthundersvm.so', '../include/thundersvm/svm_interface_api.h')
        end
        calllib('libthundersvm', 'thundersvm_predict', length(str2), str2)
    else
        disp 'OS not supported!'
    end
