Search.setIndex({docnames:["apis/athena","apis/athena.datasets","apis/athena.experiment","apis/athena.layers","apis/athena.models","apis/athena.solvers","apis/athena.tuning","apis/athena.utils","apis/athena.visualizations","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["apis\\athena.rst","apis\\athena.datasets.rst","apis\\athena.experiment.rst","apis\\athena.layers.rst","apis\\athena.models.rst","apis\\athena.solvers.rst","apis\\athena.tuning.rst","apis\\athena.utils.rst","apis\\athena.visualizations.rst","index.rst"],objects:{"athena.datasets":{base:[1,0,0,"-"],cifar10:[1,0,0,"-"],mnist:[1,0,0,"-"]},"athena.datasets.base":{BaseDataset:[1,1,1,""],DataLoaderBuilder:[1,1,1,""]},"athena.datasets.base.BaseDataset":{builder:[1,2,1,""],default_test_transform:[1,2,1,""],default_train_transform:[1,2,1,""]},"athena.datasets.base.DataLoaderBuilder":{batch_sampler:[1,2,1,""],batch_size:[1,2,1,""],build:[1,2,1,""],collate_fn:[1,2,1,""],download:[1,2,1,""],drop_last:[1,2,1,""],num_workers:[1,2,1,""],pin_memory:[1,2,1,""],root:[1,2,1,""],sampler:[1,2,1,""],shuffle:[1,2,1,""],target_transform:[1,2,1,""],test:[1,2,1,""],timeout:[1,2,1,""],train:[1,2,1,""],transform:[1,2,1,""],use_default_transforms:[1,2,1,""],worker_init_fn:[1,2,1,""]},"athena.datasets.cifar10":{cifar10:[1,1,1,""]},"athena.datasets.cifar10.cifar10":{builder:[1,2,1,""],default_test_transform:[1,2,1,""],default_train_transform:[1,2,1,""],mean:[1,3,1,""],std:[1,3,1,""]},"athena.datasets.mnist":{mnist:[1,1,1,""]},"athena.datasets.mnist.mnist":{builder:[1,2,1,""],default_test_transform:[1,2,1,""],default_train_transform:[1,2,1,""],mean:[1,3,1,""],std:[1,3,1,""]},"athena.experiment":{Experiment:[2,1,1,""],ExperimentBuilder:[2,1,1,""],Experiments:[2,1,1,""],ExperimentsBuilder:[2,1,1,""]},"athena.experiment.Experiment":{builder:[2,2,1,""],fit_one_cycle:[2,2,1,""],get_model:[2,2,1,""],get_solver:[2,2,1,""],gradcam_misclassified:[2,2,1,""],lr_range_test:[2,2,1,""],plot_lr_finder:[2,2,1,""],plot_misclassified:[2,2,1,""],plot_scalars:[2,2,1,""],run:[2,2,1,""]},"athena.experiment.ExperimentBuilder":{build:[2,2,1,""],data:[2,2,1,""],epochs:[2,2,1,""],force_restart:[2,2,1,""],log_directory:[2,2,1,""],name:[2,2,1,""],props:[2,2,1,""],resume_from_checkpoint:[2,2,1,""],solver:[2,2,1,""],train_loader:[2,2,1,""],trainer_args:[2,2,1,""],val_loader:[2,2,1,""]},"athena.experiment.Experiments":{builder:[2,2,1,""],plot_scalars:[2,2,1,""],run:[2,2,1,""]},"athena.experiment.ExperimentsBuilder":{add:[2,2,1,""],build:[2,2,1,""],get_log_dir:[2,2,1,""],handle:[2,2,1,""],log_directory:[2,2,1,""],name:[2,2,1,""]},"athena.layers.depthwiseconv":{DepthwiseConv2d:[3,1,1,""]},"athena.layers.ghostbatchnorm":{GhostBatchNorm:[3,1,1,""]},"athena.models.cifar10_v1":{Cifar10V1:[4,1,1,""]},"athena.models.cifar10_v2":{Cifar10V2:[4,1,1,""]},"athena.models.davidnet":{DavidNet:[4,1,1,""]},"athena.models.mnist":{MnistNet:[4,1,1,""]},"athena.solvers":{base_solver:[5,0,0,"-"],classification_solver:[5,0,0,"-"]},"athena.solvers.base_solver":{BaseSolver:[5,1,1,""]},"athena.solvers.base_solver.BaseSolver":{configure_optimizers:[5,2,1,""],forward:[5,2,1,""],get_lr_log_dict:[5,2,1,""],training:[5,3,1,""]},"athena.solvers.classification_solver":{ClassificationSolver:[5,1,1,""]},"athena.solvers.classification_solver.ClassificationSolver":{acc_fn:[5,2,1,""],on_train_epoch_start:[5,2,1,""],training:[5,3,1,""],training_epoch_end:[5,2,1,""],training_step:[5,2,1,""],validation_epoch_end:[5,2,1,""],validation_step:[5,2,1,""]},"athena.tuning.lr_finder":{DataLoaderIter:[6,1,1,""],ExponentialLR:[6,1,1,""],LRFinder:[6,1,1,""],LinearLR:[6,1,1,""],TrainDataLoaderIter:[6,1,1,""],ValDataLoaderIter:[6,1,1,""]},"athena.tuning.lr_finder.DataLoaderIter":{dataset:[6,2,1,""],inputs_labels_from_batch:[6,2,1,""]},"athena.tuning.lr_finder.ExponentialLR":{get_lr:[6,2,1,""]},"athena.tuning.lr_finder.LRFinder":{plot:[6,2,1,""],range_test:[6,2,1,""],reset:[6,2,1,""]},"athena.tuning.lr_finder.LinearLR":{get_lr:[6,2,1,""]},"athena.utils":{transforms:[7,0,0,"-"]},"athena.utils.transforms":{ToNumpy:[7,4,1,""],ToTensor:[7,4,1,""],UnNormalize:[7,1,1,""]},"athena.visualizations":{gradcam:[8,0,0,"-"],misclassified:[8,0,0,"-"],scalars:[8,0,0,"-"]},"athena.visualizations.gradcam":{GradCam:[8,1,1,""],GradCamPP:[8,1,1,""],apply_gradcam:[8,4,1,""],gradcam_misclassified:[8,4,1,""],overlay_gradcam_mask:[8,4,1,""],plot_gradcam:[8,4,1,""]},"athena.visualizations.gradcam.GradCam":{forward:[8,2,1,""]},"athena.visualizations.gradcam.GradCamPP":{forward:[8,2,1,""]},"athena.visualizations.misclassified":{plot_misclassified:[8,4,1,""]},"athena.visualizations.scalars":{plot_scalars:[8,4,1,""]},athena:{experiment:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"008":4,"024":4,"100":[2,4,6],"10000":2,"101":4,"102":4,"103":4,"104":4,"105":4,"106":4,"107":4,"108":4,"109":4,"110":4,"111":4,"112":4,"113":4,"114":4,"115":4,"128":[4,9],"1307":1,"170":4,"192":4,"1994":1,"1e4":2,"201":1,"2010":1,"2023":1,"256":4,"280":4,"300":6,"304":4,"3081":1,"320":4,"32x32":1,"380":4,"4465":1,"448":4,"4822":1,"4914":1,"512":4,"560":4,"640":4,"646":4,"650":4,"728":4,"792":4,"8x8":1,"911":4,"abstract":[],"boolean":6,"case":[],"class":[1,2,3,4,5,6,7,8,9],"default":[1,2,3,4,5,6,8],"final":[2,6],"float":[1,2,4,5,6,8],"function":[1,2,5,6,7,8,9],"import":[6,9],"int":[1,2,3,4,6,8],"new":[],"return":[1,2,5,6,7,8,9],"static":2,"super":2,"true":[1,2,6,8,9],"while":[2,8],Axes:6,One:2,That:6,The:[0,1,2,3,4,5,6,7,8,9],Then:5,There:2,These:[],Used:1,Uses:[],Using:2,Will:[2,6],With:[],__init__:2,__iter__:6,_lrschedul:[5,6],_train_loss:6,abc:[],about:8,acc:[2,6],acc_fn:[2,5,6],acc_lr_find:6,access:2,accumul:[2,6],accumulation_step:[2,6],accuraci:[2,6],achar:4,acquir:6,act:[],activ:8,actual:8,add:[2,6,9],add_experi:[],add_metr:[],added:[],addit:2,after:[1,2,6,9],against:8,albument:1,algorithm:8,all:[1,2,5],alpha:8,alreadi:2,also:8,altern:6,although:9,alwai:1,amount:[2,8],amplitud:2,ani:[2,6],anneal:2,anneal_strategi:2,anyth:7,apart:[],api:[1,2,5],appli:[1,2,8],apply_gradcam:8,approach:6,arg:2,argument:[2,5],around:2,arrai:7,assertionerror:[],assign:[4,9],assum:6,asynchron:[2,6],attach:[],attribut:[],auto_reset:6,avail:6,averag:[],avgpool2d:4,axes:6,axi:2,backward:4,bar:[],base:[0,2,5,6,7,8],base_dataset:[],base_lr:2,base_momentum:2,base_solv:[0,2],basedataset:1,basesolv:[2,5],batch:[1,2,3,4,5,6,9],batch_data:6,batch_idx:5,batch_sampl:1,batch_siz:[1,6,9],batchnorm2d:4,batchresult:[],been:2,befor:[1,2,8],begin:2,behind:2,being:8,best_loss:[2,6],between:[1,2,6],bool:[1,2,4,5,6,8],both:8,boundari:[2,6],build:[1,2,9],buildabl:2,builder:[1,2,5,9],built:2,bundl:2,cach:6,cache_dir:6,calcul:6,call:[1,2,6],callabl:[1,2,5,6,8],caller:[],can:[2,6,9],chain:[],channel:[3,4],check:[],checkpoint:2,child:6,cifa10_v1:0,cifa10_v2:0,cifar10:0,cifar10_test_transform:[],cifar10_train_transform:[],cifar10_v1:4,cifar10_v2:4,cifar10v1:4,cifar10v2:4,cifar:[1,4],class_idx:8,class_label:[2,8],classif:[2,5],classifi:[],classification_solv:[0,8],classificationsolv:[2,5,8,9],classificationsolverbuild:[],classmethod:1,cleanup:[],close:[],cls:2,cmap:8,code:[3,8],collate_fn:1,collect:1,combin:[],commonli:1,compil:2,complet:[],compos:1,comput:[2,6,8],configure_optim:[2,5],consist:[],constraint:5,constructor:[2,5],contain:[1,4,5,6,7,8],conv2d:4,conveni:[],converg:2,convert:[2,6,7,8],convolut:3,copi:1,core:[2,5,9],correctli:[],cos:2,cosin:2,count:[],cours:4,cpu:[2,6,8],creat:[1,2,6,9],criterion:[2,6],crop:1,cross:5,cuda:[1,2,6,9],current:6,current_checkpoint_path:[],curv:6,custom:[2,9],custom_loss_fn:9,custom_value_1:2,custom_value_2:2,cutout:1,cycl:2,cycle_momentum:2,cyclic:6,data:[1,2,6,8,9],data_load:[6,8],dataload:[1,2,6,8],dataloaderbuild:1,dataloaderit:6,datalod:[],dataset:[0,2,4,6,8,9],dataset_cl:1,davidnet:0,decor:[],def:[2,6,9],default_loss_fn:[],default_test_transform:1,default_train_transform:1,defin:[1,2,5,6,9],degre:1,delet:2,depth:3,depthwiseconv2d:[3,4],depthwiseconv:0,describ:2,desired_b:6,determin:2,develop:9,devic:[2,6,8,9],dict:[2,5,6],dictionari:[],differ:6,dilat:3,directori:[1,2,6,8],disabl:[2,6],displai:[],distribut:6,div_factor:2,diverge_th:[2,6],divid:[],document:8,doesn:7,don:5,done:[],download:1,draw:[1,2],drop:1,drop_last:1,dropout:4,dropout_valu:4,dure:[4,9],each:[1,2,5,6],easi:2,either:[],element:[],els:9,enabl:[],encapsul:[],end:2,end_lr:[2,6],entropi:5,epoch:[2,6,9],estim:4,etc:[],eva:4,evalu:6,everi:2,exampl:[6,9],except:[2,8],exclus:1,execut:[],exp:[2,6,9],expect:[],experi:[0,5,8,9],experimentbuild:2,experimentsbuild:2,explicitli:2,exponenti:[2,6],exponentiallr:6,extend:[],extens:9,factor:[2,6],fals:[1,2,4,6,8],fast:2,fastai:6,featur:3,figsiz:[2,8],figur:[2,6,8],file:6,final_div_factor:2,finder:6,first:6,fit:2,fit_one_cycl:2,flag:[1,6],flip:1,flush:[],flush_print:[],forc:2,force_restart:2,form:1,forward:[2,4,5,8],from:[1,2,3,5,6,8,9],func:[],gamma:[2,9],gener:[6,8],get:[5,8],get_acc_fn:[],get_devic:[],get_epoch:[],get_histori:[],get_log_dir:2,get_loss_fn:[],get_lr:6,get_lr_log_dict:5,get_metr:[],get_metric_nam:[],get_misclassifi:[],get_model:2,get_optim:[],get_progbar:[],get_schedul:[],get_solv:2,get_start_epoch:[],get_test_load:[],get_train_load:[],getter:2,ghost:[2,3,4,9],ghostbatchnorm:0,given:[2,6,8],good:1,gpu:6,gradcam:[0,2],gradcam_misclassifi:[2,8],gradcampp:8,gradient:[2,6,8],gradient_accumul:6,graph:[6,8],gray_r:8,group:[2,5],guess:6,hand:[],handl:2,happen:2,has:[2,5],has_metr:[],hasn:2,have:[5,8],heatmap:[2,8],help:8,helper:9,here:[3,8],highest:8,histori:[],horizont:1,how:[1,2,6],howev:6,ignor:6,imag:[2,8],implement:[1,3,4,5],implicitli:6,implment:4,in_channel:[3,4],includ:[],incomplet:1,increas:6,index:[2,6,8],indic:1,info:8,inform:[2,6],inherit:6,initi:[2,6],initial_lr:2,input:[1,3,4,6,7,8],input_data:[],inputs_labels_from_batch:6,inspect:6,instanc:[],instead:[2,4],interfac:2,interpret:8,interv:[2,6],invers:2,invok:2,is_avail:9,item:[],iter:[1,2,6],its:1,itself:6,job:2,join:2,just:[2,5,6],kbar:[],keep:[],kei:2,kera:[],kernel:3,kernel_s:3,keyword:[],kwarg:[1,7],label:[1,2,6,8],labl:6,larg:2,larger:6,last:[1,2,6],last_epoch:[2,6],latest:[],layer:[0,2,4,8],learn:[2,5,6],least:2,lesli:6,less:8,lightingmodul:5,lightn:[2,5],lightningmodul:[5,8],like:[1,2,6],likelihood:[],line:[2,6],linear:[2,6],linearli:6,linearlr:6,list:[1,2,6,7],load:1,load_state_dict:[],loader:[2,6],loader_it:6,log:[2,8,9],log_dir:[2,8],log_dir_par:[],log_directori:[2,9],log_lr:[2,6],log_result:[],log_softmax:9,logarithm:6,longer:6,look:8,loss:[2,5,6,9],loss_fn:[5,9],lower:[2,6],lr_find:6,lr_finder:0,lr_range_test:2,lr_schedul:[6,9],lrfinder:6,lrschedul:[],made:[4,9],make:2,manag:[],mani:[1,2],manner:6,manual:6,map:[1,8],mask:[2,8],matplotlib:[2,6,8],max:[],max_at_epoch:2,max_checkpoints_to_keep:[],max_lr:2,max_momentum:2,max_to_keep:[],maximum:6,maxpool2d:4,mean:[1,2,7,8],memori:[1,2,6],memory_cach:6,merg:1,messag:2,method:[2,5,6],metric:[],min_lr:2,mini:1,minim:6,minimum:2,misclassifi:[0,2],mnist:[0,2,9],mnist_test_transform:[],mnist_train_transform:[],mnistnet:[2,4,9],mode:[2,6],model:[0,2,3,5,6,8,9],modul:[0,2],momentum:[2,9],more:[2,6,8],morev:5,move:[2,6],mulitpl:[],multi:6,multipl:9,must:[1,5],mutual:1,mycustomsolv:2,name:[2,5,9],ndarrai:7,need:2,need_one_hot:6,neg:1,net:6,network:[2,6],neural:[2,6],nll:[],nll_loss:9,non:[1,4,6],non_blocking_transf:[2,6],none:[1,2,5,6,8],norm:[2,4,9],normal:[1,2,3,6,7],note:[2,6],notebook:[],notic:6,now:6,num_featur:3,num_it:[2,6],num_split:[3,4,9],num_work:1,number:[1,2,3,4,6,8],numpi:7,obj:2,object:[1,2,6,7,8],obtain:[],occur:6,old:[1,2],older:[],on_train_epoch_start:5,one:[2,6,9],onli:[2,6],onto:[],opac:[2,8],oper:7,optim:[2,5,6,9],optimizer_cl:[],option:[1,2,3,4,5,6,8],order:5,ordereddict:2,ordin:6,other:[2,6,9],otherwis:[2,6],out:6,out_channel:3,output:[3,4,5,6,8],over:6,overlai:[2,8],overlaid:8,overlap:[],overlay_gradcam_mask:8,overriden:6,packag:9,pad:[1,3],pair:2,paper:2,param:[4,5],paramet:[1,2,3,4,5,6,7,8,9],parent:2,partial:6,pass:[4,5,6],path:[1,2,6,8],pattern:[1,2],peak:2,per:[],percentag:4,perform:[1,2,3,6,7,8],phase:[],pin:[1,2,6],pin_memori:1,pkbar:[],place:6,plot:[2,6,8],plot_experi:[],plot_gradcam:8,plot_lr_find:2,plot_misclassifi:[2,8],plot_mod:2,plot_scalar:[2,8],plu:8,point:6,polici:[2,6],posit:1,possibl:[2,6],practic:6,pre:6,precis:6,predict:[],prepar:6,print:2,probabl:[],problem:5,process:[],produc:6,prog_bar:[],prog_bar_upd:[],program:9,progress:[],prop:[2,9],prope:8,properti:[2,6],provid:[2,6],pytorch:[1,2,5,6,8],pytorch_lightn:5,quickli:9,rais:[2,6,7,8],random:1,rang:[2,6],range_test:6,rapidli:9,rate:[2,5,6],read:2,readi:[],real_b:6,reciev:[],record:[],redefin:6,reduct:[],refer:6,regard:[2,8],regim:[],regress:[],regression_solv:[],regressionsolv:[],regular:[4,5],relu:4,remain:2,repres:[2,6],request:8,requir:6,reset:6,resnet32:[],resnet:9,restart:[2,6],restor:6,restrict:5,result:[2,6,8],resum:2,resume_from_checkpoint:2,retain:8,retain_graph:8,revers:7,root:1,rotat:1,rtype:[],run:[2,6,9],running_correct:[],running_process:[],running_train_loss:[],same:[5,6],sampl:1,sampler:1,save:[2,6,8],save_path:[2,8],scalar:[0,2],scale:[2,6],schedul:[2,5,9],scheduler_cl:[],school:9,scratch:2,second:[],section:2,seed:1,self:[2,6],send:8,set:[1,2,6,9],set_experi:[],set_log_dir:[],set_loss_fn:[],set_progbar:[],setter:[],setup:6,sgd:[2,9],shape:[4,7,8],should:[1,2,6,8],should_use_tqdm:[],show_lr:[2,6],shown:6,shuffl:[1,6],shyamant:4,side:1,signifi:[],significantli:6,similarli:6,simpl:[4,5],sinc:2,singl:[5,8],situat:9,size:[1,2,3,4,6,8,9],skip:2,skip_end:[2,6],skip_start:[2,6],smith:6,smooth:[2,6],smooth_f:[2,6],solver:[0,2,8,9],solver_cl:[],some:[1,3],sourc:[1,2,3,4,5,6,7,8],specif:2,specifi:[1,2,6,9],spell:5,split:[2,3,4,9],srikanth:4,standard:6,start:[2,6],start_lr:[2,6],state:6,state_dict:6,statement:[],std:[1,2,7,8],stdout:2,steepest:[2,6],step:[2,6],step_mod:[2,6],step_siz:[2,9],steplr:[2,9],stepresult:[],still:[],stop:[2,6],store:[2,6],str:[1,2,5,6,8],strategi:[1,2],string:[2,6,8],stuff:2,style:[1,2],subclass:2,subpackag:3,subprocess:1,suggest:[2,6],suggest_lr:[2,6],summari:4,summarywrit:[],support:6,sure:[],surpass:[2,6],syntax:6,system:6,tag:[],take:6,taken:[3,8],target:[1,2,5,8],target_lay:[2,8],target_transform:1,temporari:6,tenorboard:2,tensor:[1,2,5,6,7,8],tensorboard:8,test:[1,2,6,9],test_load:[2,9],test_step:[],than:8,thei:6,them:[],thi:[1,2,3,5,6,7,8,9],thomwolf:6,three:2,threshold:[2,6],through:6,tht:2,thu:[],time:1,timeout:1,tip:6,tonumpi:7,torch:[1,5,6,7,8,9],torch_lr_find:6,torchvis:1,total:[2,4],totensor:7,tqdm:[],track:[],train:[1,2,5,6],train_arg:[],train_data:6,train_data_it:6,train_dl:6,train_load:[2,6,9],train_on_batch:[],train_step:[],trainabl:4,traindataloaderit:6,trainer:2,trainer_arg:2,training_epoch_end:5,training_step:5,trainit:6,trainload:6,tranform:1,transform:[0,1,8],transpar:[],tri:[2,6],trim:6,tune:[0,2],tupl:[2,3,6,8],two:6,type:[1,2,4,5,6,7,8],typic:[],under:[2,6],union:[1,2,3],unnorm:[2,7,8],updat:2,upper:2,use:[1,2,3,4,5,6,8,9],use_default_transform:[1,9],use_ghost_batch_norm:[2,4,9],use_gradcampp:8,use_tqdm:[],used:[1,2,3,4,5,6,8,9],user_featur:6,user_histori:6,uses:[2,6,8],using:[1,2,5,6,8],util:[0,1,6,9],val_load:[2,6,9],valdataloaderit:6,valid:[2,7,8],validation_epoch_end:5,validation_step:5,valu:[1,2,6],valuabl:6,valueerror:[2,6,7],variou:[0,1,4,7,8,9],verbos:2,veri:2,vertic:[2,6],via:2,vision:9,visual:[0,2,6],wai:6,want:[1,2,5,6],weight:8,well:[1,6,9],what:6,when:[1,2,6,8],where:[2,6],whether:[1,2,4,8],which:[1,2,5,6,8,9],wide:6,wise:3,within:[0,2,6],won:[],work:[2,6],worker:1,worker_init_fn:1,wrap:[2,6],writer:[],writer_add_model:[],writer_add_scalar:[],writer_clos:[],y_label:6,y_pred:9,y_true:9,you:[5,6,9],your:6,zipsi:4},titles:["athena package","athena.datasets package","athena.experiment package","athena.layers package","athena.models package","athena.solvers package","athena.tuning package","athena.utils package","athena.visualizations package","Welcome to Athena\u2019s documentation!"],titleterms:{"function":[],athena:[0,1,2,3,4,5,6,7,8,9],base:1,base_dataset:[],base_solv:5,checkpoint:[],cifa10_v1:4,cifa10_v2:4,cifar10:1,classification_solv:5,dataset:1,davidnet:4,depthwiseconv:3,document:9,experi:2,ghostbatchnorm:3,gradcam:8,histori:[],index:9,layer:3,lr_finder:6,misclassifi:8,mnist:[1,4],model:4,modul:[1,3,4,5,6,7,8],packag:[0,1,2,3,4,5,6,7,8],regression_solv:[],scalar:8,solver:5,subpackag:0,transform:7,tune:6,usag:9,util:7,visual:8,welcom:9}})