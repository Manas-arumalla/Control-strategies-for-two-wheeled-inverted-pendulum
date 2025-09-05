m_w = 0.432;      
m_p = 5.0;        
I_w = 1/2 * (0.432) * (0.0726)^2;
I_p = 5 * (0.4)^2;                    
l   = 0.4;      
r   = 0.0726;   
g   = 9.81;     

a = 2*m_w + m_p + 2*I_w/r^2;    
b = m_p * l;
c = m_p * l^2 + I_p;
d = m_p * g * l;
Delta = a*c - b^2; 

A = [0, 1,        0,       0;
     0, 0, -b*d/Delta,    0;
     0, 0,        0,       1;
     0, 0,  a*d/Delta,    0];

B = [0;
     c/Delta;
     0;
    -b/Delta];

C = [1, 0, 0, 0; 
     0, 0, 1, 0];  

D = [0; 0];

sys = ss(A, B, C, D);

% Introduce a known input delay (for example, tau = 0.1 sec)
tau = 0.1;
sys_delay = ss(A, B, C, D, 'InputDelay', tau);

% Obtain a 2nd-order Pade approximation of the delay
sys_pade = pade(sys_delay, 2);  % This returns an approximated system with additional states

% Extract the augmented state-space matrices
[A_aug, B_aug, C_aug, D_aug] = ssdata(sys_pade);

% For example, assume 6 states (4 original + 2 from the delay):
Q = diag([9638.556, 160.203, 7751.573, 799.062, 1, 1]);  
R = 0.003;

K = lqr(A_aug, B_aug, Q, R);

disp('LQR gain matrix K for the augmented system:');
disp(K);

Ac = A_aug - B_aug * K;
sys_cl = ss(Ac, B_aug, C_aug, D_aug);

x0 = [0; 0; 0.1; 0; 0; 0];
t = 0:0.01:5;
[y, t, x] = initial(sys_cl, x0, t);

figure;
subplot(2,1,1);
plot(t, y(:,1), 'b', 'LineWidth',1.5);
ylabel('Cart Position (m)');
title('Closed-Loop Response (LQR with Delay Approximation)');
grid on;

subplot(2,1,2);
plot(t, y(:,2), 'r', 'LineWidth',1.5);
ylabel('Pendulum Angle (rad)');
xlabel('Time (s)');
grid on;
