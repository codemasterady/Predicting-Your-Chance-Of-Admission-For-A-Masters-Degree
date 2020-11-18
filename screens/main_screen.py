# Importing the libraries
from tkinter import *
from models.neural_engine import NeuralEngine


# Function to get the following command
def secondaryGUI(score):
    sub_root = Tk()
    sub_root.title("Your Prediction Is !!!!")
    canvas = Canvas(sub_root, width = 500, height = 500, bg = '#5620d4')
    canvas.pack()
    label = Label(canvas, text=f'Your Score Is:\n{str(int(score))}', font=("Courier", 50), bg = '#5620d4', fg='white')
    label.pack()
    sub_root.mainloop()
    
# Function to execute the command
def execute():
    # Getting the values
    GRE = gre_entry.get()
    TOEFL = toefl_entry.get()
    UNIVERSITY_RATING = uni_rate_entry.get()
    STATEMENT_OF_PURPOSE_RATING = sop_entry.get()
    LETTER_OF_RECOMMENDATION_RATING = lor_entry.get()
    GPA = gpa_entry.get()
    RESEARCH_EXPERIENCE = r_e_entry.get()

    # Cleaning all the inputs
    GRE = int(GRE)
    TOEFL = int(TOEFL)
    UNIVERSITY_RATING = int(UNIVERSITY_RATING)
    STATEMENT_OF_PURPOSE_RATING = int(STATEMENT_OF_PURPOSE_RATING)
    LETTER_OF_RECOMMENDATION_RATING = int(LETTER_OF_RECOMMENDATION_RATING)
    GPA = int(GPA)
    RESEARCH_EXPERIENCE = int(RESEARCH_EXPERIENCE)
    
    # Deducing the final array
    returning_list = [GRE, TOEFL, UNIVERSITY_RATING, STATEMENT_OF_PURPOSE_RATING, LETTER_OF_RECOMMENDATION_RATING, GPA, RESEARCH_EXPERIENCE]
    
    # Defining the neural network
    # Initializing the neural engine object
    engine = NeuralEngine()
    # Getting the pred
    engine.mainTrain()
    chances_pred = engine.predictChances(returning_list)
    # Initializing the final output
    final_output = 0
    # Exaluating the outliers
    if chances_pred > 0.82:
        lower_bound = chances_pred - 0.18
        final_output = (chances_pred + lower_bound)/2
    elif chances_pred < 0.18:
        upper_bound = chances_pred + 0.18
        final_output = (chances_pred + upper_bound)/2
    else:
        lower_bound = chances_pred - 0.18
        upper_bound = chances_pred + 0.18
        final_output = (upper_bound + lower_bound + chances_pred)/3
    final_output = final_output - 0.60
    # Computing the score
    score = 0
    if final_output<0.3:
        score = final_output//3
    else:
        score = 10
    print(f"The Score Is {score}\nThe final output is {final_output}\nThe Predicted Probability {chances_pred}")
    # Executing the secondary GUI
    secondaryGUI(score)
        
 

# Defining the main GUI
root = Tk()
root.title("Predict Your Chances")
canvas = Canvas(master=root, width=500, height=600, bg='#cd17ff')
canvas.pack()
master_label = Label(
    master=canvas, text="Enter your scores to predict your chances", bg='#00c3ff', font=("Aerial", 10))
master_label.place(relx=0.2, rely=0, relwidth=0.6, relheight=0.1)
execute_button = Button(master=canvas, text="Execute",
                        fg='white', bg='#ff0059', command=execute)
execute_button.place(relx=0.3, rely=0.9, relheight=0.1, relwidth=0.4)
#! Left Labels
gre_label = Label(
    master=canvas, text="Graduate Record\nExaminations Score", font=("Courier", 10), bg='#cd17ff', fg='white')
gre_label.place(relx=0, rely=0.1, relwidth=0.5, relheight=0.1)
toefl_label = Label(
    master=canvas, text="TOEFL Score", font=("Courier", 10), bg='#cd17ff', fg='white')
toefl_label.place(relx=0, rely=0.2, relwidth=0.5, relheight=0.1)
uni_rate_label = Label(
    master=canvas, text="How Would You\nRate This University?\n(0-5)", font=("Courier", 10), bg='#cd17ff', fg='white')
uni_rate_label.place(relx=0, rely=0.3, relwidth=0.5, relheight=0.1)
sop_label = Label(
    master=canvas, text="How Would You\nRate Your Statement of Purpose?\n(0-5)", font=("Courier", 10), bg='#cd17ff', fg='white')
sop_label.place(relx=0, rely=0.4, relwidth=0.5, relheight=0.1)
lor_label = Label(
    master=canvas, text="How Would You\nRate Your Letter Of\nRecommendation?\n(0-5)", font=("Courier", 10), bg='#cd17ff', fg='white')
lor_label.place(relx=0, rely=0.5, relwidth=0.5, relheight=0.1)
gpa_label = Label(
    master=canvas, text="What Is Your GPA\n(0-10)", font=("Courier", 10), bg='#cd17ff', fg='white')
gpa_label.place(relx=0, rely=0.6, relwidth=0.5, relheight=0.1)
r_e_label = Label(
    master=canvas, text="Do You Have\nResearch Experience\n(0 - No), (1 - Yes)", font=("Courier", 10), bg='#cd17ff', fg='white')
r_e_label.place(relx=0, rely=0.7, relwidth=0.5, relheight=0.1)
#! Right Texts
gre_entry = Entry(master=canvas, font=("Aerial", 30))
gre_entry.place(relx=0.5, rely=0.1, relwidth=0.5, relheight=0.1)
toefl_entry = Entry(master=canvas, font=("Aerial", 30))
toefl_entry.place(relx=0.5, rely=0.2, relwidth=0.5, relheight=0.1)
uni_rate_entry = Entry(master=canvas, font=("Aerial", 30))
uni_rate_entry.place(relx=0.5, rely=0.3, relwidth=0.5, relheight=0.1)
sop_entry = Entry(master=canvas, font=("Aerial", 30))
sop_entry.place(relx=0.5, rely=0.4, relwidth=0.5, relheight=0.1)
lor_entry = Entry(master=canvas, font=("Aerial", 30))
lor_entry.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.1)
gpa_entry = Entry(master=canvas, font=("Aerial", 30))
gpa_entry.place(relx=0.5, rely=0.6, relwidth=0.5, relheight=0.1)
r_e_entry = Entry(master=canvas, font=("Aerial", 30))
r_e_entry.place(relx=0.5, rely=0.7, relwidth=0.5, relheight=0.1)
root.mainloop()
