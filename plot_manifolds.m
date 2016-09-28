function plotit(i)
filename=strcat('xfile_2_',num2str(i),'.tna');
xfile=load(filename);
filename=strcat('yfile_2_',num2str(i),'.tna');
yfile=load(filename);
filename=strcat('cfile_2_',num2str(i),'.tna');
cfile=load(filename);

inirow=1;
AlphaData=zeros(size(cfile(inirow:end,:)));
AlphaData(1,:)=1;
AlphaData(end,:)=1;

%surf(xfile,yfile,cfile,'AlphaData',AlphaData,'FaceColor','red','FaceAlpha',0.7,'FaceLighting','gouraud','EdgeAlpha','flat')
surf(xfile(inirow:end,:),yfile(inirow:end,:),cfile(inirow:end,:),'EdgeColor','black','AlphaData',AlphaData,'EdgeAlpha','flat','FaceColor','red','FaceAlpha',0.5,'MeshStyle','row')
%surf(xfile(2:end,:),yfile(2:end,:),cfile(2:end,:))
%surf(xfile,yfile,cfile,'FaceColor','red','FaceAlpha',0.7,'FaceLighting','gouraud','EdgeColor','none')
