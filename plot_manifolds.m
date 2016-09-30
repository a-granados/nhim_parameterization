function plot_manifolds
for angl=0:5:360
    hold off;
    iterunstable=10;
    iterstable=10;
    for i=0:iterunstable
        filename=strcat('xfile-unstable_2_',num2str(i),'.tna');
        xfile=load(filename);
        filename=strcat('yfile-unstable_2_',num2str(i),'.tna');
        yfile=load(filename);
        filename=strcat('cfile-unstable_2_',num2str(i),'.tna');
        cfile=load(filename);
        inirow=1;
        AlphaData=zeros(size(cfile(inirow:end,:)));
        AlphaData(1,:)=1;
        AlphaData(end,:)=1;
        CData=zeros(size(cfile,1),size(cfile,2),3);
        CData(1,:,1)=1;
        CData(end,:,1)=1;
        h1=surf(xfile(inirow:end,:),yfile(inirow:end,:),cfile(inirow:end,:),'CData',CData,'EdgeColor','flat','AlphaData',AlphaData,'EdgeAlpha','flat','FaceColor','red','FaceAlpha',0.1,'MeshStyle','row');
        
        hold on;
    end
    for i=0:iterstable
        filename=strcat('xfile-stable_2_',num2str(i),'.tna');
        xfile=load(filename);
        filename=strcat('yfile-stable_2_',num2str(i),'.tna');
        yfile=load(filename);
        filename=strcat('cfile-stable_2_',num2str(i),'.tna');
        cfile=load(filename);
        AlphaData=zeros(size(cfile(inirow:end,:)));
        AlphaData(1,:)=1;
        AlphaData(end,:)=1;
        CData=zeros(size(cfile,1),size(cfile,2),3);
        CData(1,:,2)=1;
        CData(end,:,2)=1;
        h1=surf(xfile(inirow:end,:),yfile(inirow:end,:),cfile(inirow:end,:),'CData',CData,'EdgeColor','flat','AlphaData',AlphaData,'EdgeAlpha','flat','FaceColor','green','FaceAlpha',0.1,'MeshStyle','row');

    end
    %%Now we plot the manifold itself:
    Kvals=load('Kvals_2_filtered.tna');
    hold on;
    nr=400;
    nc=200;
    for i=1:nr
        for j=1:nc
            xKvals(i,j)=Kvals((i-1)*nc+j,1);
            yKvals(i,j)=Kvals((i-1)*nc+j,2);
            cKvals(i,j)=Kvals((i-1)*nc+j,3);
        end
    end
    CData=zeros(size(cKvals,1),size(cKvals,2),3);
    CData(1,:,1)=1;
    CData(end,:,3)=1;
    AlphaData=zeros(size(cKvals));
    AlphaData(:,1)=1;
    AlphaData(:,end)=1;
    h1=surf(xKvals,yKvals,cKvals,'EdgeColor','black','AlphaData',AlphaData,'EdgeAlpha','flat','FaceColor','interp','FaceAlpha',0.5,'MeshStyle','column');
    %axis([-1.5 1.5 -0.6 0.5 0 1.2])
    xlabel('\fontsize{36}{0}$x$','Interpreter','LaTex');
    ylabel('\fontsize{36}{0}$y$','Interpreter','LaTex');
    zlabel('\fontsize{36}{0}$c$','Interpreter','LaTex');
    ax=gca;
    ax.FontSize=20;
    view(angl,angl);    
    pause
end
