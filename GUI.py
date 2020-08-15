"""
Created on Sun August  2 14:09:42 2020
@author: Maheen Saleem
Registration No: SP17-BSE-023
"""



import numpy as np
from tkinter import *
from AI_project_Decision_Tree import *
from AI_project_Random_forest import *
from AI_Project_with_kfold import *




def main(*args):
    option = variable.get()
    print(option)
    if(option=='clf_entropy'):

        b1 = Button(frame, text = 'Predict', command =main)
        b1.grid(row=15, column=2)
        clf_entropy_()
        
    elif(option=='clf_gini'):
        
        b1 = Button(frame, text = 'Predict', command = main)
        b1.grid(row=15, column=2)
        clf_gini_()
    elif(option=='Random Forest'):
        
        e13.grid(row=12,column=2)
        label_14.config(text="Select number of decision Trees")
        b1 = Button(frame, text = 'Predict', command = random_Forest)
        b1.grid(row=15, column=2)
  

def clf_gini_():
    battery = int(e1.get())
    four_g = int(e2.get())
    int_mem = int(e3.get())
    n_cores = int(e4.get())
    ram = int(e5.get())
    talk_time = int(e6.get())
    touch_screen = int(e7.get())
    
    output, accuracy = clf_gini(battery, four_g, int_mem, n_cores, ram, talk_time, touch_screen, X_train, y_train, X_test)
    
    if(output==0):
        output='200$'
    
      
    elif(output==1):
        output='350$'
    
        
    elif(output==2):
        output='500$'
        
    elif(output==3):
        output='750$'
    
    label_16.config(text=clf_gini_kfold(X,y))
    label_11.config(text=output)
    label_13.config(text=accuracy_score(y_test,accuracy)*100)
    

def clf_entropy_():
    battery = int(e1.get())
    four_g = int(e2.get())
    int_mem = int(e3.get())
    n_cores = int(e4.get())
    ram = int(e5.get())
    talk_time = int(e6.get())
    touch_screen = int(e7.get())
    
    output, accuracy = clf_entropy(battery, four_g, int_mem, n_cores, ram, talk_time, touch_screen, X_train, y_train, X_test)
    
    if(output==0):
        output='200$'
    
      
    elif(output==1):
        output='350$'
    
        
    elif(output==2):
        output='500$'
        
    elif(output==3):
        output='750$'
    
    label_16.config(text=clf_entropy_kfold(X,y))
    label_11.config(text=output)
    label_13.config(text=accuracy_score(y_test,accuracy)*100)

def random_Forest():
    battery = int(e1.get())
    four_g = int(e2.get())
    int_mem = int(e3.get())
    n_cores = int(e4.get())
    ram = int(e5.get())
    talk_time = int(e6.get())
    touch_screen = int(e7.get())
    
    model = RandomForestClassifier(n_estimators=int(e13.get()), 
                               bootstrap = True,
                               max_features = 'sqrt')
    
    
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
#    print(X_test)
    X=np.array([[battery, four_g, int_mem, n_cores, ram, talk_time, touch_screen]])

    output = model.predict(X)
    
    print(output)
    if(output[0]==0):
        output='200$'
    
      
    elif(output[0]==1):
        output='350$'
    
        
    elif(output[0]==2):
        output='500$'
        
    elif(output[0]==3):
        output='750$'
    
    
    label_11.config(text=output)
    label_13.config(text=accuracy_score(y_test,predictions)*100)
    

    kfold_result = random_forest_kfold(AX,y,100)

    label_16.config(text=kfold_result)
    

    e13.delete(0,END)
    e13.grid_remove()
    label_14.config(text="")
    b1 = Button(frame, text = 'Predict', command = main)
    b1.grid(row=15, column=2)





root = Tk()
root.title("Mobile Price prediction")
root.configure(background='#FDE79C')             



frame = Frame(root, height = 480, width =1000)
frame.grid(row=0)
frame.config(background = '#FDE79C', padx=200,pady=50)

             

variable = StringVar(root)
variable.set("Select") # default value



label_1 = Label(frame,text ='Select algorithm for prediction: ')
label_1.config(background = '#FDE79C',width=30)
label_1.grid(row=1, column=1)



w = OptionMenu(frame, variable, "clf_gini", "clf_entropy", "Random Forest")

    
w.config(background = '#FDE79C',width=20)
w.grid(row=1, column=2,padx=50)


label_2 = Label(frame,text ='Battery Power(mAH): ')
label_2.config(background = '#FDE79C',width=30)
label_2.grid(row=3, column=1)
label_2.grid(pady=15)



label_3 = Label(frame,text ='4g(0 or 1): ')
label_3.config(background = '#FDE79C',width=30)
label_3.grid(row=4, column=1)
label_3.grid(pady=15)



label_4 = Label(frame,text ='Internal Memory(GB): ')
label_4.config(background = '#FDE79C',width=30)
label_4.grid(row=5, column=1)
label_4.grid(pady=15)



label_5 = Label(frame,text ='Number of Cores: ')
label_5.config(background = '#FDE79C',width=30)
label_5.grid(row=6, column=1)
label_5.grid(pady=15)



label_6 = Label(frame,text ='Ram(MB): ')
label_6.config(background = '#FDE79C',width=30)
label_6.grid(row=7, column=1)
label_6.grid(pady=15)




label_7 = Label(frame,text ='Talk Time(Hours): ')
label_7.config(background = '#FDE79C',width=30)
label_7.grid(row=8, column=1)
label_7.grid(pady=15)




label_8 = Label(frame,text ='Touch Screen(0 or 1): ')
label_8.config(background = '#FDE79C',width=30)
label_8.grid(row=9, column=1)
label_8.grid(pady=15)



e1 = Entry(frame, width = 26)
e1.grid(row=3,column=2)
e1.grid(pady=15)



e2 = Entry(frame, width = 26)
e2.grid(row=4,column=2)
e2.grid(pady=15)



e3 = Entry(frame, width = 26)
e3.grid(row=5,column=2)
e3.grid(pady=15)


e4 = Entry(frame, width = 26)
e4.grid(row=6,column=2)
e4.grid(pady=15)



e5 = Entry(frame, width = 26)
e5.grid(row=7,column=2)
e5.grid(pady=15)



e6 = Entry(frame, width = 26)
e6.grid(row=8,column=2)
e6.grid(pady=15)



e7 = Entry(frame, width = 26)
e7.grid(row=9,column=2)
e7.grid(pady=15)



label_9 = Label(frame,text ='Predicted price: ')
label_9.config(background = '#FDE79C',width=30)
label_9.grid(row=10, column=1)
label_9.grid(pady=15)



label_10 = Label(frame,text ='Predicted Price: ')
label_10.config(background = '#FDE79C',width=30)
label_10.grid(row=10, column=1)
label_10.grid(pady=15)



label_11 = Label(frame,text ='')
label_11.config(background = '#FDE79C',width=30)
label_11.grid(row=10, column=2)
label_11.grid(pady=10)



label_12 = Label(frame,text ='Accuracy against X_test:')
label_12.config(background = '#FDE79C',width=30)
label_12.grid(row=11, column=1)
label_12.grid(pady=10)



label_13 = Label(frame,text ='')
label_13.config(background = '#FDE79C',width=30)
label_13.grid(row=11, column=2)
label_13.grid(pady=10)



label_14 = Label(frame, text="")
label_14.config(background = '#FDE79C',width=30)
label_14.grid(row=12,column=1)
label_14.grid(pady=10)



e13 = Entry(frame, width = 26)


b1 = Button(frame, text = 'Predict', command = main)
b1.grid(row=15, column=2)


label_15 = Label(frame, text="Accuracy using K-fold cross validation:" )
label_15.config(background = '#FDE79C',width=30)
label_15.grid(row=13,column=1)

label_16 = Label(frame, text="" )
label_16.config(background = '#FDE79C',width=30)
label_16.grid(row=13,column=2)

root.mainloop()