function svm_train_matlab(a)
    if not(libisloaded('libthundersvm'))
        loadlibrary('../build/lib/libthundersvm', '../include/thundersvm/svm_matlab_interface.h')
    end
	str = {'thundersvm-train'}
	str2 = [str a]
	calllib('libthundersvm', 'thundersvm_train_matlab', length(str2), str2)
end
