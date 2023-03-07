import os
from helper import  *

def main():
    cwd = os.getcwd()

    inf = load_transform_data(path='DZ_Consumption_price_index.csv')
    # print(inf.shape) (300, 1) 
    # print(inf.columns) CPI(%)
    if not os.path.exists(os.path.join(cwd, "results")):
      os.mkdir(cwd + "\\results") 
    if not os.path.exists(os.path.join(cwd+'\\results', "graphs")):
        os.mkdir(cwd + "\\results\\graphs")
    graghs_path = cwd + "\\results\\graphs"

    plot_description(time_series=inf, title='Algerian Inflation rate from 1998 to 2022 (y.o.y)',path = 'description_plot.png')

    plot_decomposition(time_series= inf,var_name='Inflation_rate',path = 'decomposition_plot.png')

    plot_heatmap(time_series=inf, name='CPI(%)',path ='cpi_inflation_heatmap.png')

    data_split(time_series=inf)

    plot_description(time_series=inf,title='Inflation rate (Train,Test,Vlidation)',path='training_dataset.png',if_split=True)
                 

if __name__ == "__main__":
    main()              