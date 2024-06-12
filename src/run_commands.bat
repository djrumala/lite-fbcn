echo Running training for fold 1
python train.py --round R1


echo Running testing for fold 1
python test.py --model mobilenetv1 --sampling_name R1
python test.py --model mobilenetv2 --sampling_name R1
python test.py --model efficientnet --sampling_name R1


echo Running training for fold 2
python train.py --round R2

echo Running testing for fold 2
python test.py --model mobilenetv1 --sampling_name R2
python test.py --model mobilenetv2 --sampling_name R2
python test.py --model efficientnet --sampling_name R2

echo Running training for fold 3
python train.py --round R3

echo Running testing for fold 3
python test.py --model mobilenetv1 --sampling_name R3
python test.py --model mobilenetv2 --sampling_name R3
python test.py --model efficientnet --sampling_name R3


echo Running training for fold 4
python train.py --round R4

echo Running testing for fold 4
python test.py --model mobilenetv1 --sampling_name R4
python test.py --model mobilenetv2 --sampling_name R4
python test.py --model efficientnet --sampling_name R4

echo Running training for fold 5
python train.py --round R5

echo Running testing for fold 5
python test.py --model mobilenetv1 --sampling_name R5
python test.py --model mobilenetv2 --sampling_name R5
python test.py --model efficientnet --sampling_name R5

echo All tasks completed.
pause
