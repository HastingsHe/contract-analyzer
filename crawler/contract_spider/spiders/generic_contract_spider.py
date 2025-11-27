import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from contract_spider.items import ContractItem
import re

class GenericContractSpider(CrawlSpider):
    name = 'generic_contract'
    allowed_domains = ['example.com', 'contracts.com', 'legal.com']
    start_urls = [
        'https://example.com/contracts',
        'https://contracts.com/samples',
        'https://legal.com/contract-templates'
    ]

    # 规则：提取链接并跟进，同时处理合同页面
    rules = (
        # 跟进所有包含contract或agreement的链接
        Rule(LinkExtractor(allow=r'(contract|agreement|template)', deny=r'(login|register|admin)'), 
             callback='parse_contract', follow=True),
        # 跟进分页链接
        Rule(LinkExtractor(allow=r'(page|p=|page=)', deny=r'(login|register|admin)'), follow=True),
    )

    def parse_contract(self, response):
        """解析合同页面"""
        # 检查页面是否包含合同内容
        contract_content = self.extract_contract_content(response)
        
        if contract_content:
            # 提取合同标题
            title = response.css('h1::text').get() or response.css('h2::text').get() or 'Untitled Contract'
            title = title.strip() if title else 'Untitled Contract'
            
            # 提取合同类型
            contract_type = self.extract_contract_type(title, contract_content)
            
            # 创建合同项
            item = ContractItem()
            item['title'] = title
            item['url'] = response.url
            item['content'] = contract_content
            item['contract_type'] = contract_type
            item['source'] = response.url.split('/')[2]  # 提取域名作为来源
            item['crawled_at'] = scrapy.utils.misc.now()
            
            yield item

    def extract_contract_content(self, response):
        """从页面中提取合同内容"""
        # 尝试不同的选择器提取合同内容
        selectors = [
            # 常见的合同内容容器
            'div[class*="contract"], div[class*="agreement"], div[class*="content"]',
            'article',
            'main',
            'section',
            # 如果以上都不行，尝试提取所有段落
            'body'
        ]
        
        for selector in selectors:
            content_elements = response.css(selector)
            if content_elements:
                # 提取文本内容
                text_content = ' '.join(content_elements.css('*::text').getall())
                # 清理文本
                text_content = self.clean_contract_content(text_content)
                # 检查内容长度，只返回足够长的内容
                if len(text_content) > 500:  # 合同内容至少500字符
                    return text_content
        
        return None

    def clean_contract_content(self, content):
        """清理合同内容"""
        # 移除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        # 移除广告和无关内容
        content = re.sub(r'(广告|Advertisement|Sponsored|赞助|推广).*?(?=\n|$)', '', content, flags=re.IGNORECASE)
        # 移除导航和菜单文本
        content = re.sub(r'(首页|Home|关于我们|About|联系我们|Contact|服务|Services).*?(?=\n|$)', '', content, flags=re.IGNORECASE)
        return content.strip()

    def extract_contract_type(self, title, content):
        """从标题和内容中提取合同类型"""
        # 常见合同类型
        contract_types = [
            '服务合同', 'Service Agreement',
            '销售合同', 'Sales Contract',
            '租赁合同', 'Lease Agreement',
            '雇佣合同', 'Employment Contract',
            '保密协议', 'NDA', 'Non-Disclosure Agreement',
            '合作协议', 'Partnership Agreement',
            '许可协议', 'License Agreement',
            '采购合同', 'Purchase Agreement',
            '代理合同', 'Agency Agreement',
            '技术合同', 'Technology Agreement'
        ]
        
        # 检查标题中的合同类型
        for contract_type in contract_types:
            if contract_type.lower() in title.lower():
                return contract_type
        
        # 检查内容中的合同类型
        for contract_type in contract_types:
            if contract_type.lower() in content.lower():
                return contract_type
        
        return '其他合同'
