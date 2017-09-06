#!/usr/bin/env bash
#trainingData="dataset/letter.scale dataset/satimage.scale dataset/usps dataset/acoustic_scale "
filename="dataset/satimage.scale"
#for filename in $trainingData
#do
	#C=0.5
	C=0.5
	for (( i=0; i<15;i++ ))
	do
		C=$(echo "2*$C"|bc)
		gamma=1
		for(( j=0; j<10; j++))
		do
			gamma=$(echo "scale=10; $gamma * 2 "|bc)
			echo $C
			echo $gamma
			echo $filename
			./svm-train -c $C -g $gamma -h 0 $filename ;
		done
	done
#done
