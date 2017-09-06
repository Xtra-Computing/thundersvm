#!/usr/bin/env bash
#trainingData="dataset/letter.scale dataset/satimage.scale dataset/usps dataset/acoustic_scale "
filename="dataset/pendigits"
#for filename in $trainingData
#do
	#C=0.5
	C=32
	for (( i=0; i<1;i++ ))
	do
		C=$(echo "2*$C"|bc)
		gamma=2048
		for(( j=0; j<30; j++))
		do
			gamma=$(echo "scale=10; $gamma / 2 "|bc)
			echo $C
			echo $gamma
			echo $filename
			./svm-train -c $C -g $gamma  $filename ;
		done
	done
#done
