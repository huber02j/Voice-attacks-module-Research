"# Voice-attacks-module-Research" 

finalprojectscript.py is designed intentially for the data collection "ReMASC: Realistic Replay Attack Corpus for Voice Controlled Systems". Therefore, in order to use it will any set of voice recordings, it will need to be organized in the right way so that the program can distinguish between training, test, and validation files. The naming conventions of the files might also need to be modified so that the program can determine the different 'events' for the machine learning model. 

For the upscalling, toggle the variables in line 117 and 118.
- 0 0 for no upscalling
- 1 0 for random upscalling
- 0 1 for smote upscalling
