
    HOW TO RUN OUR PROJECT:
        #1 - If there is any "build" directory -> delete "build" directory;

        #2 - Then, at the terminal, run the following commands:
            mkdir build
            cd build
            cmake ..
            make
            ../CV_Project

        #3 - It will run through all the test images at folder Food_leftover_dataset
            and will save the resulting boxes and masks at the respective folders at
            FoodResults;

        #4 - All the necessary metrics are calculated in the end and saved inside
        FoodResults as well.