import time
from requests.exceptions import RequestException
from xml.dom import minidom
import requests
import xml.etree.ElementTree as ET
from typing import Callable, List, Dict, Optional
from functools import wraps
from dataclasses import dataclass

from .API import API

@dataclass
class ArticleData:
    pmid: str
    title: str
    abstract: str
    is_book: bool
    book_title: Optional[str] = None

    def __str__(self):
        if self.is_book:
            return f"PMID: {self.pmid}\nBook Title: {self.book_title}\nChapter Title: {self.title}\nAbstract: {self.abstract}"
        else:
            return f"PMID: {self.pmid}\nTitle: {self.title}\nAbstract: {self.abstract}"


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
    def __init__(self, proxy=None):
        super().__init__()
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.proxy = proxy

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
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "retmax": 1000,
        }

        all_results = ET.Element("PubmedArticleSet")

        for i in range(0, len(pmids), 1000):
            batch = pmids[i:i+1000]
            params["id"] = ",".join(batch)

            try:
                if len(batch) > 200:
                    response = requests.post(self.base_url, data=params, proxies={'http': self.proxy, 'https': self.proxy} if self.proxy else None)
                else:
                    response = requests.get(self.base_url, params=params, proxies={'http': self.proxy, 'https': self.proxy} if self.proxy else None)
                # print(f"Request URL: {response.url}")
                # # print the request url with parameters so that we can see what is being requested
                # print(f"Request URL: {response.url}{response.request.body}")
                # print(f"Response status code: {response.status_code}")
                # # Print first 500 characters
                # print(f"Response content: {response.content[:500]}...")

                batch_xml = ET.fromstring(response.content)
                xml_str = minidom.parseString(
                    ET.tostring(batch_xml)).toprettyxml(indent="  ")

                # check if its a PubMedArticle or a PubmedBookArticle
                if batch_xml.find(".//PubmedArticle") is not None:
                    all_results.extend(batch_xml.findall(".//PubmedArticle"))
                elif batch_xml.find(".//PubmedBookArticle") is not None:
                    all_results.extend(
                        batch_xml.findall(".//PubmedBookArticle"))

                # all_results.extend(batch_xml.findall(".//PubmedArticle"))
            except Exception as e:
                print(f"Error processing batch {i}-{i+999}: {str(e)}")
                print(f"PMIDs in this batch: {batch}")

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
                is_book=False
            )
            article_data_list.append(article_data)

        for book in xml.findall('.//PubmedBookArticle'):
            pmid = book.find('.//PMID').text
            book_elem = book.find('.//BookDocument')

            book_title = book_elem.find('.//BookTitle')
            chapter_title = book_elem.find('.//ArticleTitle')
            abstract = book_elem.find('.//AbstractText')

            article_data = ArticleData(
                pmid=pmid,
                title=chapter_title.text if chapter_title is not None else "Chapter title not available",
                abstract=abstract.text if abstract is not None else "Abstract not available",
                is_book=True,
                book_title=book_title.text if book_title is not None else "Book title not available"
            )
            article_data_list.append(article_data)

        return article_data_list
