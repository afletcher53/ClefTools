import time
from requests.exceptions import RequestException
from xml.dom import minidom
import requests
import xml.etree.ElementTree as ET
from typing import Callable, List, Dict
from functools import wraps
from dataclasses import dataclass

from Classes.API import API


@dataclass
class ArticleData:
    pmid: str
    title: str
    # authors: List[str]
    # journal: str
    # # pubdate: str
    # doi: str
    abstract: str

    def __str__(self):
        return f"PMID: {self.pmid}\nTitle: {self.title}\nAuthors: {', '.join(self.authors)}\nJournal: {self.journal}\nPublication Date: {self.pubdate}\nDOI: {self.doi}\nAbstract: {self.abstract}"


def retry_on_exception(max_retries=3, backoff_factor=0.3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RequestException as e:
                    wait = backoff_factor * (2 ** retries)
                    print(f"Request failed. Retrying in {wait:.2f} seconds...")
                    time.sleep(wait)
                    retries += 1
            raise Exception(
                "Max retries reached. Unable to complete the request.")
        return wrapper
    return decorator


class PubMed(API):
    def __init__(self):
        super().__init__()
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def error_guard(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # xml_str = minidom.parseString(
            #     ET.tostring(result)).toprettyxml(indent="  ")

            if result.find('.//ERROR') is not None:
                raise Exception(f"Error: {result.find('.//ERROR').text}")
            # if result.find('.//AbstractText') is None or result.find('.//AbstractText').text == "":
            #     # print the pmid that does not have an abstract
            #     print(args)

            #     raise Exception(f"No abstract found for the given PMID(s)")
            return result
        return wrapper

    @error_guard
    @retry_on_exception(max_retries=3, backoff_factor=0.3)
    def get(self, pmids: List[str]) -> ET.Element:
        """
        Get full records for a list of PubMed IDs
        :param pmids: List of PubMed IDs
        :return: XML Element Tree
        """
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "retmax": 1000,  # Increased to 1000 to reduce number of requests
        }

        all_results = ET.Element("PubmedArticleSet")

        for i in range(0, len(pmids), 1000):
            batch = pmids[i:i+1000]

            if len(batch) > 200:
                params["id"] = ",".join(batch)
                response = requests.post(self.base_url, data=params)
            else:
                params["id"] = ",".join(batch)
                response = requests.get(self.base_url, params=params)

            batch_xml = ET.fromstring(response.content)
            all_results.extend(batch_xml.findall(".//PubmedArticle"))

        return all_results

    def get_abstract(self, pmid: str) -> str:
        """
        Get the abstract for a single PubMed ID
        :param pmid: PubMed ID
        :return: Abstract if it exists, otherwise a message indicating it's not available
        """
        xml = self.get([pmid])
        abstract = xml.find('.//AbstractText')
        return abstract.text if abstract is not None else "Abstract not available"

    def get_title(self, pmid: str) -> str:
        """
        Get the title for a single PubMed ID
        :param pmid: PubMed ID
        :return: Title
        """
        xml = self.get([pmid])
        return xml.find('.//ArticleTitle').text

    def get_authors(self, pmid: str) -> List[str]:
        """
        Get the authors for a single PubMed ID
        :param pmid: PubMed ID
        :return: List of authors
        """
        xml = self.get([pmid])
        authors = xml.findall('.//Author')
        return [f"{author.find('LastName').text}, {author.find('ForeName').text}" for author in authors]

    def get_journal(self, pmid: str) -> str:
        """
        Get the journal for a single PubMed ID
        :param pmid: PubMed ID
        :return: Journal
        """
        xml = self.get([pmid])
        return xml.find('.//Journal/Title').text

    def get_pubdate(self, pmid: str) -> str:
        """
        Get the publication date for a single PubMed ID
        :param pmid: PubMed ID
        :return: Publication date
        """
        xml = self.get([pmid])
        pub_date = xml.find('.//PubDate')
        return f"{pub_date.find('Year').text}-{pub_date.find('Month').text}-{pub_date.find('Day').text}"

    def get_doi(self, pmid: str) -> str:
        """
        Get the DOI for a single PubMed ID
        :param pmid: PubMed ID
        :return: DOI
        """
        xml = self.get([pmid])
        doi = xml.find(".//ArticleId[@IdType='doi']")
        return doi.text if doi is not None else "DOI not available"

    def get_attributes(self, pmid: str, attributes: List[str]) -> Dict[str, str]:
        """
        Get the specified attributes for a single PubMed ID
        :param pmid: PubMed ID
        :param attributes: List of attributes
        :return: Dictionary of attributes
        """
        xml = self.get([pmid])
        result = {}
        for attr in attributes:
            element = xml.find(f'.//{attr}')
            result[attr] = element.text if element is not None else f"{attr} not available"
        return result

    def get_all(self, pmid: str) -> Dict[str, str]:
        """
        Get all metadata for a single PubMed ID
        :param pmid: PubMed ID
        :return: Dictionary of metadata
        """
        xml = self.get([pmid])
        return {child.tag: child.text for child in xml.iter() if child.text and child.text.strip()}

    @retry_on_exception(max_retries=3, backoff_factor=0.3)
    def get_article_data_from_list(self, article_pmids: List[str]) -> List[ArticleData]:
        """
        Get all metadata for a list of PubMed IDs using a single API call
        :param article_pmids: List of PubMed IDs
        :return: List of ArticleData objects
        """
        xml = self.get(article_pmids)
        article_data_list = []

        for article in xml.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text
            article_elem = article.find('.//Article')

            title = article_elem.find('.//ArticleTitle')
            abstract = article_elem.find('.//AbstractText')

            article_data = ArticleData(
                pmid=pmid,
                title=title.text if title is not None else "Title not available",
                abstract=abstract.text if abstract is not None else "Abstract not available",
            )
            article_data_list.append(article_data)

        return article_data_list
