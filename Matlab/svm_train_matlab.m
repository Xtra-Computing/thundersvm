function svm_train_matlab(a)
    if not(libisloaded('libthundersvm'))
        if ispc
            loadlibrary('../build/bin/Debug/thundersvm.dll', '../include/thundersvm/svm_interface_api.h')
        elseif ismac
            loadlibrary('../build/lib/libthundersvm.dylib', '../include/thundersvm/svm_interface_api.h')
        elseif isunix
            loadlibrary('../build/lib/libthundersvm.so', '../include/thundersvm/svm_interface_api.h')
        else
            disp 'OS not supported!'
        end
    end
	str = {'thundersvm-train'}
	str2 = [str a]
	calllib('libthundersvm', 'thundersvm_train', length(str2), str2)
end
