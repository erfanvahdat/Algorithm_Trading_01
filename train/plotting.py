def plot_result(data,result):

    fig=plt.figure(figsize=(25  ,7))
    ax=fig.add_subplot(1,1,1)
    plt_candle(data=data,ax=ax,period=18,path_data=False,full=True,ich=False)

    alpha= data.Close.std() - 0.01
    
    plt.scatter(np.array(result['two_pairs'])[:,0][:,1],np.array(result['two_pairs'])[:,0][:,0] -  alpha,c='green',marker='^',label='swing_low')
    plt.scatter(np.array(result['two_pairs'])[:,1][:,1],np.array(result['two_pairs'])[:,1][:,0] + alpha ,c='red',marker='v',label='swing_high')

    # Lines
    for i in result['Draw_line']:
        plt.plot(i[1],i[0])


    # Enter Pricex  
    enter_price,enter_date=result['EnterPrice'][-1][1],result['EnterPrice'][-1][0]

    
    i=result['postion_inf'][-1]
    sl_price=i[0]
    tp_price=i[1]
    tp2_price=i[2]
    r1=i[3]
    r2=i[-1]

    date_refrence=np.array(result['two_pairs'])[0,1,1]
    time_extend= datetime.timedelta(hours=30)
    time_extend_inf= datetime.timedelta(hours=20)

    font = {'family' : 'Arial',
            'weight' : 'bold',
            'size'   : 10}


    # Enter candle
    # for i in np.array(result['EnterPrice']):

    plt.vlines(x=np.array(result['EnterPrice'])[:,0] , ymax=np.array(result['EnterPrice'])[:,1]+ alpha, ymin=np.array(result['EnterPrice'])[:,1] )             
        
    # ax.annotate(text='Enter',xy=(enter_date,enter_price - alpha*alpha) ,arrowprops=dict(arrowstyle= '<|-|>',
    #                         color='blue',
    #                         lw=1,
    #                         ls='-'),**font,xytext=(enter_date,  enter_price + alpha*2),label='Enter')
                                    
    # First sl
    # ax.hlines(y=sl_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='#990000', linestyle='solid',linewidth=2)
    # plt.annotate(text=f'Sl -> {sl_price :.2f}',xy=(date_refrence + time_extend_inf , sl_price) ,**font)

    # # First tp
    # ax.hlines(y=tp_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='green', linestyle='dashed')
    # plt.annotate(text=f'tp_1 -> {tp_price :.2f}',xy=(date_refrence + time_extend_inf , tp_price) ,**font)
    
    # # Second tp
    # ax.hlines(y=tp2_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='green', linestyle='solid',)
    
    # plt.annotate(text=f'tp_2 -> {tp2_price :.2f}',xy=(date_refrence + time_extend_inf , tp2_price), **font)

    # # Enter point
    # ax.hlines(y=enter_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='#D8D056', linestyle='solid',label='Enter')
    
    # plt.annotate(text=f'Enter point -> {enter_price :.2f}',xy=(date_refrence + time_extend_inf , enter_price), **font)
    
    

    ax.set_title(loc='Center',label='Test',font=font)
    # ax.gca().set_title('title')
    ax.legend(loc='upper right',prop={'size':10})
    
    # plt.savefig('plotting.jpeg')
    plt.plot()
    
# plot_result(data=data,result=result)




