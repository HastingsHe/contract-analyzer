import scrapy
from scrapy.loader.processors import MapCompose, TakeFirst
from w3lib.html import remove_tags

class ContractItem(scrapy.Item):
    """合同数据项"""
    # 合同标题
    title = scrapy.Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )
    # 合同URL
    url = scrapy.Field(
        output_processor=TakeFirst()
    )
    # 合同内容
    content = scrapy.Field(
        input_processor=MapCompose(remove_tags, str.strip),
        output_processor=TakeFirst()
    )
    # 合同类型
    contract_type = scrapy.Field(
        output_processor=TakeFirst()
    )
    # 来源网站
    source = scrapy.Field(
        output_processor=TakeFirst()
    )
    # 爬取时间
    crawled_at = scrapy.Field(
        output_processor=TakeFirst()
    )
