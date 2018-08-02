
python -m train --dataset aflw --images_path /home/samuel/aflw/ --label detection --batch_size 100 --steps 500 --ol detection_large1 --os detection_small1 --epochs 10

python -m train --dataset wiki --images_path ~/home/samuel/wiki/wiki --label age --batch_size 100 --steps 500 --lr 1e-4  --ol detection_age_large1 --os detection_age_small1 --epochs 10;

python -m train --dataset wiki --images_path ~/mitiku/dataset/wiki/wiki/ --label gender --batch_size 100 --steps 500 --lr 1e-5  --ol detection_age_gender_large1 --os detection_age_gender_small1 --load_model models/detection_age_large1.h5 --epochs 10;

python -m train --dataset celeba --images_path ~/mitiku/dataset/img_align_celeba/ --label smile --batch_size 100 --steps 500 --lr 1e-5  --ol detection_age_gender_smile_large1\
    --os detection_age_gender_smile_small1 --load_model models/detection_age_gender_large1.h5 --epochs 10;

python -m train --dataset celeba --images_path ~/Desktop/lfw/ --label smile --batch_size 100 --steps 500 --lr 1e-5  --ol detection_age_gender_smile_large1 --os detection_age_gender_smile_small1 --load_model models/detection_age_gender_large1.h5 --epochs 10

python -m train --dataset wiki --images_path ~/dataset/wiki/ --label age --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10

python -m train --dataset wiki --images_path ~/Videos/Adience/faces/ --label age --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10

python -m train --dataset adience --images_path /home/samuel/aflw/ --label detection --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10

python -m train --dataset adience --images_path /home/samuel/aflw/ --label pose --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10			

python -m train --dataset adience --images_path /home/samuel/aflw/ --label pose --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10

python -m train --dataset adience --images_path /home/samuel/aflw/ --label pose --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10			

python -m train --dataset adience --images_path /home/samuel/dataset/adience --label pose --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10	

python -m train --dataset adience --images_path /home/samuel/Videos/adience --label pose --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10	

python -m train --dataset adience --images_path /home/samuel/Videos/adience --label detection --batch_size 100 --steps 500 --lr 1e-4 --ol detection_age_large1 --os detection_age_small --epochs 10


