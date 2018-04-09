for number in {0..9}
do
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_DeViSE_1_$number --SELATT=2 --TA=1 --maxSteps=20 --OPT=2 --numClass=32 --numAtt=300
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_DeViSE_1_$number --SELATT=2 --OPT=4 --numClass=32 --numAtt=300 > Log$number.txt
done