import time
import pandas as pd
from ortools.linear_solver import pywraplp


# StudentCourseData[i][j] = 1 if Student i wants to take Course j
# and equals 0 if Student i does not want to take Course j.
# For example, StudentCourseData[3][8] = 1

StudentCourseData = [
 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
 [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
 [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
 [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
 [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]]


# TeacherCourseData[i][j] = 1 if Teacher i can teach Course j
# and equals 0 if Teacher i cannot teach Course j.
# For example, TeacherCourseData[5][9] = 1

TeacherCourseData = [
 [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
 [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]]


# TeacherDayData[i][j] = 1 if Teacher i can teach on Day j
# and equals 0 if Teacher i cannot teach on Day j.
# For example, TeacherDayData[5][3] = 1

TeacherDayData = [
 [1, 1, 0, 0],
 [1, 1, 1, 1],
 [1, 1, 1, 0],
 [0, 1, 1, 1],
 [1, 1, 1, 1],
 [0, 0, 1, 1]]


# CourseDayData[i][j] = 1 if Course i can be offered on Day j
# and equals 0 if Course i cannot be offered on Day j.
# For example, CourseDayData[0][3] = 0

CourseDayData = [
 [1, 1, 0, 0],
 [0, 0, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]


numStudents = 8
numTeachers = 6
numCourses = 12
numDays = 4

allStudents = range(numStudents)
allTeachers = range(numTeachers)
allCourses = range(numCourses)
allDays = range(numDays)

StudentList=['S0','S1','S2','S3','S4','S5','S6','S7']
TeacherList=['T0','T1','T2','T3','T4','T5']
CourseList=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11']
DayList=['D0','D1','D2','D3']


solver = pywraplp.Solver('Timetabling Problem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

start_time = time.time()

# Define our binary variables for the students and teachers
X = {}
for s in allStudents:
    for c in allCourses:
        for d in allDays:
            X[s,c,d] = solver.BoolVar('X[%i,%i,%i]' % (s,c,d))

Y = {}
for t in allTeachers:
    for c in allCourses:
        for d in allDays:
            Y[t,c,d] = solver.BoolVar('Y[%i,%i,%i]' % (t,c,d))
            

# ****** PLEASE DO NOT MODIFY ANYTHING ABOVE THIS LINE *******
    
    
# Define our objective function - this will need to be fixed (how?)
solver.Maximize(solver.Sum(1 * X[s,c,d] for s in allStudents for c in allCourses for d in allDays))


# Each student must take one course on each day
for s in allStudents:
    for d in allDays:
        solver.Add(solver.Sum([X[s,c,d] for c in allCourses]) == 1)  

        
# No teacher may teach more than one course per day
for t in allTeachers:
    for d in allDays:
        solver.Add(solver.Sum([Y[t,c,d] for c in allCourses]) <= 1)  


### Add all other constraints that must be added!

# (ii)+(iv) Each course is taught exactly once overall
for c in allCourses:
    solver.Add(solver.Sum([Y[t,c,d] for t in allTeachers for d in allDays]) == 1)

# (iii) Exactly three courses are taught each day
for d in allDays:
    solver.Add(solver.Sum([Y[t,c,d] for t in allTeachers for c in allCourses]) == 3)

# (v) Each teacher teaches exactly two courses
for t in allTeachers:
    solver.Add(solver.Sum([Y[t,c,d] for c in allCourses for d in allDays]) == 2)

# Feasibility logic
for t in allTeachers:
    for c in allCourses:
        # teacher–course compatibility
        if TeacherCourseData[t][c] == 0:
            for d in allDays:
                solver.Add(Y[t,c,d] == 0)
for t in allTeachers:
    for d in allDays:
        # teacher available on that day
        if TeacherDayData[t][d] == 0:
            for c in allCourses:
                solver.Add(Y[t,c,d] == 0)
for c in allCourses:
    for d in allDays:
        # course allowed on that day
        if CourseDayData[c][d] == 0:
            for t in allTeachers:
                solver.Add(Y[t,c,d] == 0)

# Students may only take courses they want
for s in allStudents:
    for c in allCourses:
        if StudentCourseData[s][c] == 0:
            for d in allDays:
                solver.Add(X[s,c,d] == 0)

# Link students to offered courses
# If no teacher offers course c on day d, then no student can take it that day.
for s in allStudents:
    for c in allCourses:
        for d in allDays:
            solver.Add(X[s,c,d] <= solver.Sum([Y[t,c,d] for t in allTeachers]))
        
    
current_time = time.time() 
reading_time = current_time - start_time         
sol = solver.Solve()
solving_time = time.time() - current_time

print('Optimization Complete with Total Happiness Score of', round(solver.Objective().Value()))
print("")
print('Our program needed', round(solving_time,3), 
      'seconds to determine the optimal solution')
                

# Print Output for Students

for s in allStudents:
    for c in allCourses:
        for d in allDays:
            if X[s,c,d].solution_value() == 1:
                print("Student", StudentList[s], "is taking Course", CourseList[c],
                      "on Day", DayList[d])
    print("")


for t in allTeachers:
    for c in allCourses:
        for d in allDays:
            if Y[t,c,d].solution_value() == 1:
                print("Teacher", TeacherList[t], "is teaching Course", CourseList[c],
                      "on Day", DayList[d])
    print("")

