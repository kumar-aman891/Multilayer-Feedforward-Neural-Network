clear all;
clc;
%% Multilayer Feed Forward Neural Network


%% Part 1:Take inputs from an input file
%{
no. of input neurons(independent variables)
no. of output neurons(dependent variables)
no. of training pattetns
no. of testing patterns
%}  
fileID = fopen('input.txt','r');
ip = fscanf(fileID,'%f');
dataTraining = readmatrix('data_training');
dataTesting = readmatrix('data_testing');

%% Defining the variables
%{
P:Number of Training Patterns
L:Number of Input Neurons
I:Input Patterns Matrix(P X L)
M:Number of neurons in the Hidden Layer
N:Number of neurons in the output layer
V:Connection weight matrix between input and hidden layers
W:Connection weight matrix between hidden and output layers
T:Matrix with Target Values of the Patterns
%} 
patterns =ip(3);
I = dataTraining(1:patterns,1:8);
I1 = I';%Taking the transpose , such that each column represents a Pattern,Matrix containing input training patterns
Test_Patterns = ip(4);
I_test = (dataTesting(1:Test_Patterns,1:8))';%testing pattern matrix
M = 30; 
N = ip(2);
[L,P] = size(I1);
V = zeros(L+1,M);
W = zeros(M+1,N);

T = (dataTraining(1:patterns,9))'; %target values containing target of training patterns
T_test = (dataTesting(1:Test_Patterns,9))';%Target values containing target of testing patterns

%% Part 2: initialization of the weights between input & hidden layers ,and between hidden and output layers
%Initializing Matrix [V]
for i=1:L+1
    for j=1:M
        V(i,j) = -1+2*rand;
    end
end

%Initializing Matrix [W]
for i=1:M+1
    for j=1:N
        W(i,j) = -1+2*rand;
    end
end

%% Normalization of the input paramters
Inorm = zeros(L,P);%normalized input
for l=1:L
        minELE = min(I1(l,:));
        maxElE = max(I1(l,:));
        for p=1:P
            Inorm(l,p)=8*(I1(l,p)-minELE)/(maxElE-minELE)-4;    
        end
end

%% Normalizing the target values
minT = min(T);
maxT = max(T);
for p=1:P
    T(p)=0.1+0.8*(T(p)-minT)/(maxT-minT);
end
%% Defining calculation parameters
%{
eta : Learning rate
MSE : Mean Square Error
alpha : Momentum Coefficient
%}
eta = 0.3;
MSE=1000;
alpha =0.6;
tol =0.0013;%tolerance
count = 0;%iteration counter

%% Part 3 Training of the pattern
while(MSE>=tol)
    %{
    I_h :Matrix containing Input to hidden neurons  
    O_h :Matrix containing Output from hidden neurons
    I_O :Matrix containing Input to Output neurons
    O_O :Matrix containing output from output neurons
    delta_W & delta_V:Updating parametrs for weights.
    momentum_W & momentum_V:Momentum terms for updating of weights 
    %}
    I_h = zeros(M,P); 
    O_h = zeros(M,P);
    O_O = zeros(N,P);
    I_O = zeros(N,P);
    delta_W = zeros(M+1,N);
    delta_V = zeros(L+1,M);

   
    %% Computations and updation

    % Calculate the forward pass
    %Input to the hidden layer
  for m=1:M
      for p = 1:P
          for l=1:L
          I_h(m,p)=I_h(m,p)+Inorm(l,p)*V(l+1,m);
      end
   end
  end
  %bias
  for p=1:P
      for m=1:M
          I_h(m,p)=I_h(m,p)+1*V(1,m);
      end
  end

    %Output from the hidden layer 

    for p = 1:P
        for m=1:M
            O_h(m,p) = 1/(1+exp(-I_h(m,p)));
        end
    end

    %Input to the output layer
for n=1:N
      for p = 1:P
          for m=1:M
          I_O(n,p)=I_O(n,p)+O_h(m,p)*W(m+1,n);
      end
   end
end
    %bias
    for p=1:P
        for n=1:N
            I_O(n,p)=I_O(n,p)+1*W(1,n);
        end
    end

    %Output from the output layer

    for p = 1:P
        for n=1:N
            O_O(n,p) = 1/(1+exp(-I_O(n,p)));
        end
    end
 %% Normalizing the output values
minO = min(O_O);
maxO = max(O_O);
O_Onorm = zeros(N,P);
for p=1:P
    O_Onorm(p)=0.1+0.8*(O_O(p)-minO)/(maxO-minO);
end

    E = T-O_Onorm;%Error Matrix
    %% Calculation of delta_V and delta_W

    %Momentum term of previous iteration
    momentum_W = alpha*delta_W;
    momentum_V = alpha*delta_V;
    
    for p=1:P
        for m=1:M
            for n=1:N
                delta_W(m+1,n) = delta_W(m+1,n)+(E(n,p)*1*O_O(n,p)*(1-O_O(n,p))*O_h(m,p))*(0.8/(maxO-minO));
            end
        end
    end
    for p=1:P
            for n=1:N
                delta_W(1,n) = delta_W(1,n)+(E(n,p)*1*O_O(n,p)*(1-O_O(n,p))*1)*(0.8/(maxO-minO));
            end
    end
    
    delta_W = eta*(delta_W)*(1/P);

    for p=1:P
        for n=1:N
            for l=1:L
                for m=1:M
                  delta_V(l+1,m) = delta_V(l+1,m)+(E(n,p)*1*O_O(n,p)*(1-O_O(n,p))*W(m,n)*1*O_h(m,p)*(1-O_h(m,p))*Inorm(l,p))*(0.8/(maxO-minO));  
                end
            end
        end
    end

    for p=1:P
        for n=1:N
                for m=1:M
                  delta_V(1,m) = delta_V(1,m)+(E(n,p)*1*O_O(n,p)*(1-O_O(n,p))*W(m,n)*1*O_h(m,p)*(1-O_h(m,p))*1)*(0.8/(maxO-minO));  
                end
        end
    end

    delta_V = -eta*(delta_V)*(1/(P*N));

    % updating V & W
    W = W+delta_W+momentum_W;
    V = V+delta_V+momentum_V;

    %Caculating Mean Square Error(MSE)
    error = zeros(N,P);
    for p = 1:P
        for n = 1:N
            error(n,p) = error(n,p) + 0.5*(T(n,p)-O_Onorm(n,p))^2;
        end
    end

    sumALL =0;

    for p=1:P
        for n=1:N
            sumALL = sumALL+error(n,p);
        end
    end
    MSE = sumALL/P
    count=count+1;
    delta_error(count)=MSE;%for plotting purpose
 
end

if isnan(MSE) == 1
     disp('Out of Bounds , Please run the code again');
     return;
 end


%% Denormalized values of the output
minO = min(O_O);
maxO = max(O_O);
for p=1:P
    O_O(p)=(1/0.8)*(O_Onorm(p)-0.1)*(maxT-minT)+minT;
end

%% Plotting of graph MSE vs Iteration counter for training patterns
plot(1:count,delta_error(1,:));
xlabel('Number of iterations','fontsize',16);
ylabel('MSE','fontsize',16);
fig = gcf;
saveas(fig,'MSE_VS_ITERATIONS_TRAINING','png');

%% Part 4 Testing of the network

[L,P] = size(I_test);

%% Normalization of the input paramters
Inorm = zeros(L,P);%normalized input
for l=1:L
        minELE = min(I_test(l,:));
        maxElE = max(I_test(l,:));
        for p=1:P
            Inorm(l,p)=8*(I_test(l,p)-minELE)/(maxElE-minELE)-4;    
        end
end

%% Normalizing the target values
minT = min(T_test);
maxT = max(T_test);
for p=1:P
    T_test(p)=0.1+0.8*(T_test(p)-minT)/(maxT-minT);
end
    I_h = zeros(M,P); 
    O_h = zeros(M,P);
    O_O = zeros(N,P);
    I_O = zeros(N,P);
   
    %%  Computations

    % Calculate the forward pass
    %Input to the hidden layer
  for m=1:M
      for p = 1:P
          for l=1:L
          I_h(m,p)=I_h(m,p)+Inorm(l,p)*V(l+1,m);
      end
   end
  end
  %bias
  for p=1:P
      for m=1:M
          I_h(m,p)=I_h(m,p)+1*V(1,m);
      end
  end

    %Output from the hidden layer 

    for p = 1:P
        for m=1:M
            O_h(m,p) = 1/(1+exp(-I_h(m,p)));
        end
    end

    %Input to the output layer
for n=1:N
      for p = 1:P
          for m=1:M
          I_O(n,p)=I_O(n,p)+O_h(m,p)*W(m+1,n);
      end
   end
end
    %bias
    for p=1:P
        for n=1:N
            I_O(n,p)=I_O(n,p)+1*W(1,n);
        end
    end

    %Output from the output layer

    for p = 1:P
        for n=1:N
            O_O(n,p) = 1/(1+exp(-I_O(n,p)));
        end
    end  

    %% Normalizing the output values
minO = min(O_O);
maxO = max(O_O);
O_Onorm = zeros(N,P);
for p=1:P
    O_Onorm(p)=0.1+0.8*(O_O(p)-minO)/(maxO-minO);
end

    E = T_test-O_Onorm;

  %% Denormalized values of the output
minO = min(O_O);
maxO = max(O_O);
for p=1:P
    O_O(p)=(1/0.8)*(O_Onorm(p)-0.1)*(maxT-minT)+minT;
end

%% Calculation of relative error
%{
 R_error :Matrix containg the values for relative error between the
 predicted and actual values of the testing patterns
 %}
Target = (dataTesting(1:Test_Patterns,9))';
R_error = Target - O_O;
for i=1:P
    R_error(i)=abs(R_error(i)/Target(i));
end
%% Part 5 Writing output in the output file
fileOD = fopen('output.txt','w');
fprintf(fileOD,'%s %s %s\n','Target values  ','Predicted values  ','Relative error');
for i=1:P
    fprintf(fileOD,'%f \t\t %f \t\t %f\n',Target(i),O_O(i),R_error(i));
end

fileMSE = fopen('MSE.txt','w');
fprintf(fileMSE,'%s %s\n','MSE  ','      Iterations');
for i=1:count
    fprintf(fileMSE,'%f \t %d \n',delta_error(i),i);
end



