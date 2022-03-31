#### SIGMA: Solve Image Inpainting with  Guidance from Masked Autoencoders
-  ### Deploy Environment
    ```bash
    conda env create -f env.yaml
    pip install -r requirements.txt
    ```
    We use PyTorch=1.7.0

-  ### Download the trained models
   1. Link: https://drive.google.com/drive/folders/11hzes3BsYjwu39KDtIARPhMMNErgCEHJ?usp=sharing
   2. (optional, if you can't use the first link) link：https://rec.ustc.edu.cn/share/f8bd5d10-b0ca-11ec-aeed-4522c4269f96
   3. download the models and place them in current project's folder `./`. It should be like `./released_model/places/inp.pth`


-  ### Prepare Val/Test Data 

    1. Place the val datasets into folder structure like:
    `path_to_val/FFHQ/val/7_kinds_mask_types/000000.png`
    2. Similarily, place the test datasets into folder structure like:
    `path_to_test/FFHQ/test/7_kinds_mask_types/000000.png`
    3. The `path_to_val` and `path_to_test` is important and may be the same path.


- ### Execute Inference
    The running scripts for the inference process are in `test_script.sh` file 


