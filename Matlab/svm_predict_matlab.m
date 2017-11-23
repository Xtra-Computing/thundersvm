function svm_predict_matlab(a)
    if not(libisloaded('libthundersvm'))
        loadlibrary('../build/lib/libthundersvm', '../include/thundersvm/svm_matlab_interface.h')
    end
	str = {'thundersvm-predict'}
	str2 = [str a]
	calllib('libthundersvm', 'thundersvm_predict_matlab', length(str2), str2)
end