# demo: inference the val and test set of places dataset
test_root="/data1/xiaoqiang.zhou/datasets/ntire22/inpainting/ntire22_inp_val" # test_root/Places/val_or_test/mask_type/img
dataset_name="places"
mae_model_path="released_model/places/mae.pth"
inp_model_path="released_model/places/inp.pth"
for phase in "val" "test"
do
CUDA_VISIBLE_DEVICES=0 python test_inpaint.py \
--use_mae \
--dataset_name ${dataset_name} \
--test_root ${test_root}'/Places/'${phase} \
--mae_model_path ${mae_model_path} \
--inp_model_path ${inp_model_path} \
--output_root 'ntire_results/' \
--full_size_test \
--replace_background \
--pad \
--resize_to_ori
done


# demo: inference the val and test set of imagenet dataset
test_root="/data1/xiaoqiang.zhou/datasets/ntire22/inpainting/ntire22_inp_val" # test_root/ImageNet/val_or_test/mask_type/img
dataset_name="imagenet"
mae_model_path="released_model/imagenet/mae.pth"
inp_model_path="released_model/imagenet/inp.pth"
for phase in "val" "test"
do
CUDA_VISIBLE_DEVICES=0 python test_inpaint.py \
--use_mae \
--dataset_name ${dataset_name} \
--test_root ${test_root}'/ImageNet/'${phase} \
--mae_model_path ${mae_model_path} \
--inp_model_path ${inp_model_path} \
--output_root 'ntire_results/' \
--full_size_test \
--replace_background \
--pad \
--resize_to_ori
done

# demo: inference the val and test set of ffhq dataset
test_root="/data1/xiaoqiang.zhou/datasets/ntire22/inpainting/ntire22_inp_val" # test_root/FFHQ/val_or_test/mask_type/img
dataset_name="ffhq"
mae_model_path="released_model/ffhq/mae.pth"
inp_model_path="released_model/ffhq/inp.pth"
for phase in "val" "test"
do
CUDA_VISIBLE_DEVICES=0 python test_inpaint.py \
--use_mae \
--dataset_name ${dataset_name} \
--test_root ${test_root}'/FFHQ/'${phase} \
--mae_model_path ${mae_model_path} \
--inp_model_path ${inp_model_path} \
--output_root 'ntire_results/' \
--full_size_test \
--replace_background \
--pad \
--resize_to_ori
done

# demo: inference the val and test set of wikiart dataset
test_root="/data1/xiaoqiang.zhou/datasets/ntire22/inpainting/ntire22_inp_val" # test_root/WikiArt/val_or_test/mask_type/img
dataset_name="wikiart"
mae_model_path="released_model/wikiart/mae.pth"
inp_model_path="released_model/wikiart/inp.pth"
for phase in "val" "test"
do
CUDA_VISIBLE_DEVICES=0 python test_inpaint.py \
--use_mae \
--dataset_name ${dataset_name} \
--test_root ${test_root}'/WikiArt/'${phase} \
--mae_model_path ${mae_model_path} \
--inp_model_path ${inp_model_path} \
--output_root 'ntire_results/' \
--full_size_test \
--replace_background \
--pad \
--resize_to_ori
done