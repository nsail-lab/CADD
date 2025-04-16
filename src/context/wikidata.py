import requests
from typing import Dict, List, Union, Optional

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"

# Mapping of features to their respective Wikidata property IDs
FEATURE_TO_PROPERTY_ID = {
    'gender': 'P21',
    'birth_date': 'P569',
    'birth_place': 'P19',
    'nationality': 'P27',
    'occupation': 'P106',
    'education': 'P69',
    'spouse': 'P26',
    'children': 'P40',
    'followers': 'P8687',
}

class WikiDataRetriever:
    """Class to retrieve and process data from Wikidata."""

    def get(self, subject: str) -> Dict[str, Union[str, List[str]]]:
        """
        Retrieve data for a given subject from Wikidata.

        Args:
            subject (str): The name of the subject to retrieve data for.

        Returns:
            dict: A dictionary containing the subject's description and various features.
        """
        entity_id = self.__get_entity_id(subject)
        if not entity_id:
            return {'error': 'Entity not found'}

        data = self.__get_wikidata(entity_id)
        results = {
            'description': data['entities'][entity_id]['descriptions']['en']['value']
        }

        for feature, pid in FEATURE_TO_PROPERTY_ID.items():
            returned_feature = self.__get_wikidata_features(data['entities'][entity_id]['claims'], pid)
            if returned_feature is not None:
                results[feature] = returned_feature
        
        return results
    
    def __get_entity_id(self, name: str) -> Optional[str]:
        """
        Get the Wikidata entity ID for a given name.

        Args:
            name (str): The name to search for.

        Returns:
            str: The entity ID if found, otherwise None.
        """
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": name
        }
        response = requests.get(WIKIDATA_API_URL, params=params)
        data = response.json()

        return data['search'][0]['id'] if data['search'] else None
        
    def __get_wikidata(self, entity_id: str) -> Dict:
        """
        Get the full Wikidata entry for a given entity ID.

        Args:
            entity_id (str): The entity ID to retrieve data for.

        Returns:
            dict: The data from Wikidata.
        """
        url = WIKIDATA_ENTITY_URL.format(entity_id)
        response = requests.get(url)
        return response.json()
    
    def __get_wikidata_features(self, claims: Dict, pid: str) -> Union[str, List[str], None]:
        """
        Extract specific features from the Wikidata claims.

        Args:
            claims (dict): The claims data from Wikidata.
            pid (str): The property ID to look for.

        Returns:
            str or list: The extracted feature(s).
        """
        if pid not in claims:
            return ''

        features = []
        for claim in claims[pid]:
            feature = self.__extract_feature_from_claim(claim)
            if feature:
                features.append(feature)

        return features if len(features) > 1 else features[0] if features else None

    def __extract_feature_from_claim(self, claim: Dict) -> Optional[str]:
        """
        Extract a feature from a single claim.

        Args:
            claim (dict): A single claim from Wikidata.

        Returns:
            str: The extracted feature, or None if extraction fails.
        """
        try:
            datavalue = claim['mainsnak']['datavalue']
            value = datavalue['value']

            if datavalue['type'] == 'time':
                return value['time']
            elif datavalue['type'] == 'quantity':
                return value['amount']
            else:
                # get English label for given entity ID
                url = WIKIDATA_ENTITY_URL.format(value['id'])
                response = requests.get(url)
                entity_data = response.json()
                return entity_data['entities'][value['id']]['labels']['en']['value']
        except KeyError:
            return None

# Example usage
if __name__ == "__main__":
    retriever = WikiDataRetriever()
    subject_data = retriever.get("Albert Einstein")
    print(subject_data)