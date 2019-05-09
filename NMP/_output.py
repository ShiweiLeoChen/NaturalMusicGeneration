from NMP import output_directory
from NMP.SequenceGenerator import Generator
from termcolor import colored
import os

while True:
    try:
        epoch_id = None
        first_note = None
        length = None

        while True:
            epoch_id = int(input("Input Epoch id i (i % 10 == 0) && (10 <= i <= 90): "))
            if epoch_id % 10 != 0:
                print("Wrong epoch_id. Please try again.")
                continue
            else:
                break

        while True:
            first_note = int(input("\nInput initial note n (type(n) == int) && (0 <= n <= 86): "))
            if (first_note < 0) or (first_note > 86):
                print("Wrong first note. Please try again")
                continue
            else:
                break

        while True:
            length = int(input("\nInput length l (type(l) == int) && (l >= 300): "))
            if length < 300:
                answer = input("Length too short, errors may accrued. Do you still want to continue? (y/n) ")
                if answer == 'n':
                    continue
                else:
                    break;
            else:
                break

        print(colored("Generating Music...", "green"))
        print()

        G = Generator(epoch_id, first_note, length)
        music = G.generate_sequence()

        print(colored("Generation complete!", "green"))
        print(colored("------------------------------------------------------------", "green"))
        print(colored(music, "green"))
        print(colored("------------------------------------------------------------", "green"))

        print(colored("Writing result to disk...", "green"))
        print()

        file_name = ""
        for i in range(100000):
            file_name = "output" + str(i) + ".txt"
            file_path = output_directory + file_name
            if os.path.exists(file_path):
                continue
            else:
                with open(file_path, "w") as text_file:
                    print(music, file=text_file)
                break

        print(colored("result {} has been written to {} ".format(file_name, output_directory), "green"))
        print(colored("END---------------------------------------------------------"))
        break
    except ValueError as i:
        print(colored(("Exception type: "+str(type(i))), "red"))
        print(colored("Invalid input. Please try again.", "red"))
        print(colored("------------------------------------------------------------", "red"))
