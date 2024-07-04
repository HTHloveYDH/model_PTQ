# conda activate 8295_3.6

export SNPE_ROOT=path/to/your/QC_Snapdragon_Neural_Processing_Engine_SDK/snpe-x.x.x.x/snpe-x.x.x.x

export ANDROID_NDK_ROOT=path/to/your/QC_Snapdragon_Neural_Processing_Engine_SDK/android-ndk-x-linux/android-ndk-x

export PATH=$PATH:$SNPE_ROOT/bin/x86_64-linux-clang

echo $PATH

source bin/envsetup.sh -o /home/tinghao/anaconda3/envs/8295_3.6/lib/python3.6/site-packages/onnx

source bin/envsetup.sh -t /home/tinghao/anaconda3/envs/8295_3.6/lib/python3.6/site-packages/tensorflow