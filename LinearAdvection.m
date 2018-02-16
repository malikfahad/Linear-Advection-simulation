function LinearAdvection
    % first initialize some parameters  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the advection speed
    a=1.;
    % number of grid points in space (not counting the point at x=1, which
    %is the same as the point at x=0, due to the periodicity)
    N=51;
    % spatial step
    h=1./N;
    % safety constant for stability (should be smaller than 1)
    cfl=.8;
    % CFl timestep limit for the explicit methods
    dt=cfl*h/a;
    % time at which we want to end the simulation
    t_end=1.;
    % number of timesteps to be taken
    n_it=t_end/dt;
    % number of different methods we want to try
    n_methods=4;
        
    % initialize some arrays
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % spatial grid (not including the point at x=1, which is the same as 
    %the point at x=0, due to the periodicity)
    x=(0:h:1-h);
    % temporal grid
    t=(0:dt:n_it*dt);
    % arrays for the numerical approximations at times n and n+1
    v_new=zeros(N,n_methods);
    v_old=zeros(N,n_methods);
    % array for the exact solution
    v_exact=zeros(N,1);
    
    % the initial condition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:n_methods
        %v_old(:,i)=exp(-50*(x-.5).^2);
        v_old(:,i)=(1.+sign(x-.5))/2;
    end
    % get the matrices for the implicit method
    % A V(n+1) = B V(n)
    [A,B]=mat_linadv_BC(N,a,h,dt);
    
    [A1,B1]=mat_linadv_CN(N,a,h,dt);
    
    [A2,B2]=mat_linadv_LW(N,a,h,dt);
    % the main iteration loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for iter = 1:n_it
        % method 1: FU (treat the first point separately
        %    taking into account the periodicity)
        v_new(2:N,1)=v_old(2:N,1)-a*dt/h*(v_old(2:N,1)-v_old(1:N-1,1));
        v_new(1,1)=v_old(1,1)-a*dt/h*(v_old(1,1)-v_old(N,1));
        % method 2: the implicit method (the periodicity
        %    is taken into account in the matrices A and B)
        v_new(:,2)=A\(B*v_old(:,2));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        v_new(:,3)=A1\(B1*v_old(:,3));
        v_new(:,4)=A2\(B2*v_old(:,4));
        
        % the exact solution with a=1
        %v_exact(:) = exp(-50*(x-a*t(iter+1)-.5).^2) + ...
         %            exp(-50*(x-a*t(iter+1)+.5).^2);
        % Problem 2:
        % the exact solution with a=1 (until t=1, we need 3 unit
        % jumps in the solution...)
        v_exact(:) = (1.+sign((x-a*t(iter+1))+0.5))/2. - ...
                     (1.+sign((x-a*t(iter+1))-0.0))/2. + ...
                     (1.+sign((x-a*t(iter+1))-0.5))/2.;
                 
        % graphical output
        plot(x,v_exact(:),'^r-')
        hold on
        for i=1:n_methods
           plot(x,v_new(:,1),'*c-')
           plot(x,v_new(:,2),'-ms')
           plot(x,v_new(:,3),'ok-')
           plot(x,v_new(:,4),'+g-')
        end
        axis([0 1 -.2 1.2])
        xlabel('x')
        ylabel('v')
        title(['Linear Advection Simulation for N = ' num2str(N)''])
        hold off
        legend('exact','FU','BC','CN','LW')
        pause(0.001)
		% prepare for the next iteration
        v_old=v_new;
    end
end
% the matrices for the implicit method
%-------------------------
% implicit central in space (BC)
function [A,B]=mat_linadv_BC(N,a,h,dt)
  e = ones(N,1);
  A = spdiags([-0.5*a*dt/h*e e 0.5*a*dt/h*e], [-1 0 1], N, N);
  A(1,N)=-0.5*a*dt/h;
  A(N,1)=0.5*a*dt/h;
  B=speye(N);
end

function [A1,B1]=mat_linadv_CN(N,a,h,dt)
    e = ones(N,1);
    A1=spdiags([-e 4*h/dt*e e],[-1 0 1], N, N);
    A1(1,N)=-1;
    A1(N,1)=1;
    B1=spdiags([e 4*h/dt*e -e],[-1 0 1], N, N);
    B1(1,N)=1;
    B1(N,1)=-1;
end

function [A2,B2]=mat_linadv_LW(N,a,h,dt)
    e = ones(N,1);
    A2=speye(N);
    p=(.5*a*dt)/(h);
    q=((.5*a*a*dt*dt)/(h*h));
    B2=spdiags([(p+q)*e (1-(2*q))*e (-p+q)*e],[-1 0 1], N, N);
    B2(1,N)=p+q;
    B2(N,1)=-p+q;
end


