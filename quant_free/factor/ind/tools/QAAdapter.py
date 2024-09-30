import QUANTAXIS as QA
import tools.Sample_Tools as spl

def QA_adapter_get_blocks(hy:str):
    
    if 'sw' in hy.lower():
      a = QA.QA_fetch_sw_industry_adv(hy).data
      blocks_view = a.groupby(level=0).apply(
      lambda x:[item for item in x.index.remove_unused_levels().levels[1]])
    else:
      a = QA.QA_fetch_stock_block_adv().data
      blocks_view = a[a['type'] == hy].groupby(level=0).apply(
          lambda x:[item for item in x.index.remove_unused_levels().levels[1]]
      )

    return blocks_view

def QA_adapter_get_code_from_block(hy, block_name):
    if 'sw' in hy.lower():
      a = QA.QA_fetch_sw_industry_adv(hy, block_name).data
      codes = list(a.index.levels[1].values)
    else:
      codes = QA.QA_fetch_stock_block_adv(blockname=block_name).code

    if len(codes)!=0:
        codes = list(spl.get_codes_by_market(codes_list=codes, sse='all', only_main=True,filter_st=True))
      
    return codes
