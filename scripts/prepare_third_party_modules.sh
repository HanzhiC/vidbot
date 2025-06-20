# Clone the third-party modules
echo "Cloning third-party modules"
git submodule update --init --recursive

# EfficientSAM weight is already downloaded via cloning the EfficientSAM repo
echo "EfficientSAM weight is already downloaded via cloning the EfficientSAM repo"

# Download the weight of the GroundingDINO
echo "Downloading GroundingDINO weight"
mkdir -p third_party/GroundingDINO/weights && cd third_party/GroundingDINO/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../../

# Download the weight of the GraspNet
echo "Downloading GraspNet weight"
mkdir -p third_party/graspness_unofficial/weights && cd third_party/graspness_unofficial/weights
gdown https://drive.google.com/uc?id=10o5fc8LQsbI8H0pIC2RTJMNapW9eczqF
cd ../../../

echo "Done with preparing third-party modules"
echo "Follow their instructions for installation!"