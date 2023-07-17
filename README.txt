
    HOW TO RUN OUR PROJECT:
        #1 - If there is any "build" directory -> delete "build" directory;

        #2 - Then, at the terminal, run the following commands:
            mkdir build
            cd build
            cmake ..
            make
            ../CV_Project

        #3 - It will ask which tray to run, write one tray number (1 to 8). It will run
            through all the images of the chosen tray at folder /Food_leftover_dataset
            and will save the resulting boxes and masks at the respective folders at
            /FoodResults;

        #4 - After running manually all the trays run once again and chose "100" so the
            program will calculate all the metrics and save then inside / FoodResults
