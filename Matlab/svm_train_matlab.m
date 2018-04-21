function svm_train_matlab(a)
    if not(libisloaded('libthundersvm'))
        if ispc
            loadlibrary('../build/bin/Debug/thundersvm.dll', '../include/thundersvm/svm_matlab_interface.h')
        else
            loadlibrary('../build/lib/libthundersvm', '../include/thundersvm/svm_matlab_interface.h')
        end
    end
	str = {'thundersvm-train'}
	str2 = [str a]
	calllib('libthundersvm', 'thundersvm_train_matlab', length(str2), str2)
end
