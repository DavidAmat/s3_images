import os

# HOME DIR
DIR_SGMKR = "/home/ec2-user/SageMaker/s3_images"
PROJ_NAME = "foodi_ml"
HOME_DIR = os.path.join(DIR_SGMKR, PROJ_NAME)

# S3
S3_BUCKET = 'test-bucket-glovocds'
S3K_imgs = 'artifacts/002/'
S3_file_samples = 'training_data.csv'

# LOCAL PATHS
pth_dwn_samples = os.path.join(DIR_SGMKR, "tmp", "training_data")
pth_dwn_pictures = os.path.join(DIR_SGMKR, "tmp", "pictures")

# CSV: training_data
cols_csv = ["product_name", "collection_name", "product_descr", "image_names_s3"]
