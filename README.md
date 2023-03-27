# FEA_Solver
FEA_Solver

All functions are in solver1d, I just vary the main.py, currently works for 1D system, adding more functions to ease working on Timoshenko beams.
You can change type of element by just one click, Options are provided to use reduced integration or full integration, To facilitate that I used two main loops both runs gauss quadrature, both loops run independently allowing user to choose which type of integration one can do.

![image](https://user-images.githubusercontent.com/26081294/215250660-8e63f2dd-420f-4f3d-8d2b-683ed1e9a124.png)



Its usefull to observe effect of sheerlocking, solver1d.py is equipped to deal with bar, euler(Hermite Interpolation fns) and timmeshenko beams(QUAL and LINEAR elements). Easily apply forces and body forces, moments or boundary condition. 

### Instructions to run

- Install python 3.10.+ ( it won't work with python <3.5)
- install matplitlib and pip
- Clone the repositors
- Adjust parameters in params file and run

For we got to manually code for post processing, Main functions will just give displacements

---

a 2D plate mesh generator, result q9 element
![image](https://user-images.githubusercontent.com/26081294/227829641-c170d1f8-617f-4ed8-ba87-e478e1a3ff18.png)

