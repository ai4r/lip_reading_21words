# lip_reading_21words

## Intro

KAIST Lip reading module for HumanCare Project

## Environment
Test on Ubuntu 16.04, Python3.5.2

## Dependencies

dlib==19.6

llvm==8.0

librosa==0.7.2

opencv-python==4.2.0.34

imutils==0.5.3

tensorflow-gpu>1.12


## Getting Started


1. Download the weight files from [HERE](https://drive.google.com/drive/folders/1IceOnEoW1OQu7fANmqYhvnxb-_uaD2xx?usp=sharing) and move it in weight folder

2. Install dlib(install guide: [HERE](http://learnopencv.com/install-dlib-on-ubuntu)])


3. Install llvm8 and clang8(install guide: [HERE](https://stackoverflow.com/questions/58242715/cabbit-install-llvm-9-or-clang-0-on-ubuntu-16-04))

4. Install the requirements.txt
    ```
    pip install -r requirements.txt
    ```


## Usage - Demo

1. Lipreading Demo
    Target video has to be placed in "input_avi" folder.

    ```
    python demo_main.py
    ```
