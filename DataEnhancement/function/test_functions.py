import unittest
import pandas as pd
from unittest.mock import Mock, patch
from category import get_category
from complexity import get_complexity
from sentiment import get_sentiment
from ner import get_ner
from enrich_ne import get_enriched_ne
from party import get_party
from min_maj import get_min_maj_ratio
from story import get_story
from min_maj import get_min_maj_ratio


class TestGetCategory(unittest.TestCase):

    def test_with_candidate_labels(self):
        user_labels = ["news", "sports", "life"]
        sample_text = "In an adrenaline-charged match, the Springville Strikers snatched a thrilling win against the Riverside Rovers with a dramatic last-minute goal. The scoreboard read 2-1 in favor of the Strikers as the final whistle blew, leaving fans in awe."
        result = get_category(sample_text, candidate_labels = user_labels )
        self.assertEqual(result, "sports")

    def test_with_candidate_labels_not_found(self):
        user_labels = ["news", "sports", "life"]
        sample_text = "This text means nothing"
        result = get_category(sample_text, candidate_labels = user_labels )
        self.assertEqual(result, -1)

    def test_without_candidate_labels(self):
        meta_data = pd.DataFrame({
            "id": [1, 2, 3],
            "category": ["news", "sports", "life"]
        })
        sample_id = 3
        result = get_category(sample_id, meta_data=meta_data)
        self.assertEqual(result, "life")

    def test_without_candidate_labels_not_found(self):
        meta_data = pd.DataFrame({
            "id": [1, 2, 3],
            "category": ["news", "sports", "life"]
        })
        sample_id = 4
        result = get_category(sample_id, meta_data=meta_data)
        self.assertEqual(result, -1)


class TestGetComplexity(unittest.TestCase):

    def test_valid_text(self):
        sample_text = 'Some Supreme Court justices thought Donald Trump was setting them up. Two days after the official swearing-in of Justice Brett Kavanaugh in October 2018, the president arranged a televised ceremony at the White House and invited all the justices. Justices had declined to attend similar White House events under previous presidents, resisting the optics that would conflict with separation of powers. This time, they especially worried about being used for political purposes and were concerned that an appearance by the full contingent of sitting justices could look like an endorsement of the president. But the White House Nine Black Robes BOOK In the end, their concerns were justified. Most of the justices sat stone-faced, disturbed by what Trump said during the event and by being unwitting participants in a political exercise. Trump had a way of ensnarling the court in politics, fomenting rhetoric of personal destruction and conspiracy, all the while generating challenges to the rule of law.'            
        result = get_complexity(sample_text)# A hypothetical Flesch-Kincaid Grade Level score
        self.assertAlmostEqual(result, 13.07, delta=1e-2)

    def test_text_with_less_than_100words(self):
        text = "this is too short to compute complexity."
        result = get_complexity(text)
        self.assertEqual(result, None)


class TestGetSentiment(unittest.TestCase):

    def test_positive_sentiment(self):
        text = "This is a fantastic news article!"
        result = get_sentiment(text)
        self.assertGreater(result, 0)

    def test_negative_sentiment(self):
        text = "The article was disappointing and frustrating."
        result = get_sentiment(text)
        self.assertLess(result, 0)

    def test_neutral_sentiment(self):
        text = "This is an informative piece of writing."
        result = get_sentiment(text)
        self.assertAlmostEqual(result, 0, delta=1e-2)


class TestGetNER(unittest.TestCase):

    def test_get_ner_with_entities(self):
        sample_text = "Barack Obama visited New York. Microsoft are tech giants."
        user_entities = ['PERSON','GPE','ORG']
        result = get_ner(sample_text,entities = user_entities)
        entities_list = []

        for r in result:
            entities_list.append(r['label'])

            if r['label'] == 'PERSON':
                result_per_name = r['text']
                result_per_freq = r['frequency']
                result_per_span = r['spans']

            elif r['label'] == 'GPE':
                result_loc_name = r['text']
                result_loc_freq = r['frequency']
                result_loc_span = r['spans']

            elif r['label'] == 'ORG':
                result_org_name = r['text']
                result_org_freq = r['frequency']
                result_org_span = r['spans']

        self.assertTrue(set(entities_list) == set(user_entities))
        self.assertEqual(result_per_name , 'Barack Obama')
        self.assertEqual(result_per_freq , 1)
        self.assertEqual(result_per_span , [(0, 12)])
        self.assertEqual(result_loc_name, 'New York')
        self.assertEqual(result_loc_freq, 1)
        self.assertEqual(result_loc_span, [(21, 29)])
        self.assertEqual(result_org_name, 'Microsoft')
        self.assertEqual(result_org_freq, 1)
        self.assertEqual(result_org_span, [(31, 40)])

    def test_get_ner_with_alternative_entities(self):
        sample_text = "Barack Obama visited New York. Obama visited Microsoft in New York City. Obama was happy."
        user_entities = ['PERSON','GPE']
        result = get_ner(sample_text,entities = user_entities)

        for r in result:
            if r['label'] == 'PERSON':
                result_per_alt = r['alternative']
                result_per_fre = r['frequency']
                result_per_span = r['spans']
            elif r['label'] == 'GPE':
                result_loc_alt = r['alternative']
                result_loc_fre = r['frequency']
                result_loc_span = r['spans']

        self.assertEqual(result_per_alt, ['Barack Obama', 'Obama'])
        self.assertEqual(result_per_fre, 3)
        self.assertEqual(result_per_span , [(0, 12), (31, 36), (73, 78)])
        self.assertEqual(result_loc_alt, ['New York', 'New York City'])
        self.assertEqual(result_loc_fre, 2)
        self.assertEqual(result_loc_span, [(21, 29), (58, 71)])


class TestEnhanceNER(unittest.TestCase):

    def test_enhance_ner_found_wiki(self):
        ne_list = [{'text': 'Barack Obama', 'alternative': ['Barack Obama','Obama'], 'frequency': 1, 'label': 'PERSON'}]
        result = get_enriched_ne(ne_list)

        self.assertEqual(result[0]['Barack Obama']['givenname'], ['Barack'])
        self.assertEqual(result[0]['Barack Obama']['familyname'], ['Obama'])
        self.assertEqual(result[0]['Barack Obama']['gender'], ['male'])
        self.assertIn('politician', result[0]['Barack Obama']['occupations'])
        self.assertEqual(result[0]['Barack Obama']['party'], ['Democratic Party'])
        self.assertIn('United States of America', result[0]['Barack Obama']['citizen'])
        self.assertIn('African American', result[0]['Barack Obama']['ethnicity'])
        self.assertIn('United States of America', result[0]['Barack Obama']['place_of_birth'])
     

    def test_enhance_ner_not_found_wiki(self):
        ne_list = [{'text': 'Blair Davis', 'alternative': ['Blair Davis', 'Blair'], 'frequency': 3, 'label': 'PERSON'}]
        result = get_enriched_ne(ne_list)

        with self.assertRaises(KeyError):
            result[0]['Blair Davis']['givenname']


      
class TestGetParty(unittest.TestCase):

    def test_no_parties(self):
        ne_list = [
            {"John": {"frequency": 2}},
            {"Alice": {"frequency": 3}}
        ]
        result = get_party(ne_list)
        self.assertEqual(result, {})

    def test_single_party(self):
        ne_list = [
            {"Joe": {"frequency": 4, "party": ["Democratic Party"]}},
            {"Alice": {"frequency": 3}}
        ]
        result = get_party(ne_list)
        self.assertEqual(result, {"Democratic Party": 4})

    def test_multiple_parties(self):
        ne_list = [
            {"John": {"frequency": 2, "party": ["Republican Party"]}},
            {"Alice": {"frequency": 3}},
            {"Bob": {"frequency": 1, "party": ["Republican Party", "Independent"]}}
        ]
        result = get_party(ne_list)
        self.assertEqual(result, {"Republican Party": 3, "Independent": 1})



class TestGetMinMajRatio(unittest.TestCase):

    def test_gender_scores_with_one_item(self):
        major_genders = ['male']
        major_citizens = []
        major_ethinicities = []
        major_place_of_births = []
        ne_list = [{'Kay Hagan': {'gender': ['female'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}}, 
                   {'Elizabeth Dole': {'gender': ['male'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}}]
                        
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens, major_ethinicity=major_ethinicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['gender'], [0.5, 0.5])

    def test_gender_scores_with_multiple_item(self):
        major_genders = ['male','female']
        major_citizens = ['United States of America']
        major_ethinicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [{'Kay Hagan': {'gender': ['female'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}}, 
                   {'Elizabeth Dole': {'gender': ['male'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}}]
                        
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens, major_ethinicity=major_ethinicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['gender'], [0, 1])

    def test_ethnicity_scores_only_with_citizen(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethinicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [{'Kay Hagan': {'gender': ['female'], 'citizen': ['France'], 'ethnicity': [], 'place_of_birth': [], 'frequency': 1}}, 
                   {'Elizabeth Dole': {'gender': ['male'], 'citizen': ['Mexico'], 'ethnicity': [], 'place_of_birth': [], 'frequency': 1}},
                   {'Jackson': {'gender': ['male'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': [], 'frequency': 1}} 
                   ]                 
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens, major_ethinicity=major_ethinicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['ethnicity'], [0.6667, 0.3333])

    def test_ethnicity_scores_with_citizen_and_ethinicity(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethinicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [{'Kay Hagan': {'gender': ['female'], 'citizen': ['United States of America'], 'ethnicity': ['white people'], 'place_of_birth': [], 'frequency': 1}}, 
                   {'Elizabeth Dole': {'gender': ['male'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': [], 'frequency': 1}},
                   {'Joe Biden': {'gender': ['male'], 'citizen': ['United States of America'], 'ethnicity': ['Jewish'], 'place_of_birth': [], 'frequency': 1}}, 
                   ]                 
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens, major_ethinicity=major_ethinicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['ethnicity'], [0.3333, 0.6667])

    def test_ethnicity_scores_with_citizen_and_place_of_birth(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethinicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [{'Kay Hagan': {'gender': ['female'], 'citizen': ['United States of America'], 'ethnicity': ['white people'], 'place_of_birth': ['United States of America'], 'frequency': 1}}, 
                   {'Elizabeth Dole': {'gender': ['male'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': ['Spain'], 'frequency': 1}},
                    ]                 
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens, major_ethinicity=major_ethinicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['ethnicity'], [0.5, 0.5])


    def test_mainstream_scores(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethinicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [{'Kay Hagan': {'givenname': ['Kay'], 'gender': ['female'], 'citizen': ['United States of America'], 'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}}, 
                   {'Greensboro': {'frequency': 1}}, 
                   ]                 
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens, major_ethinicity=major_ethinicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['mainstream'], [0.5, 0.5])



class TestGetStory(unittest.TestCase):


    def test_get_story_single_category(self):
        text1 = 'The sun sets over the horizon, painting the sky with hues of orange and pink. Birds fly homeward, their silhouettes darkening against the vibrant backdrop. A sense of calm settles in as the world transitions from day to night.'
        text2 = 'As the day comes to a close, the sun dips below the edge of the earth, casting a warm glow across the landscape. Flocks of birds soar through the air, heading towards their nests. The tranquil beauty of twilight envelops everything in a serene embrace.'
        text3 = 'Beneath a sky painted with the last remnants of daylight, the sun bids its farewell, casting an amber glow over the surroundings. The air becomes cooler, and the sounds of rustling leaves and distant streams fill the atmosphere. Natures rhythm embraces the world in a soothing lullaby as night gradually takes hold.'
        sample_data = {'date': ['2019-10-01', '2019-10-02', '2019-10-03'],
                       'category': ['A', 'A', 'A'],
                       'text': [text1, text2, text3]
                        }
        sample_df = pd.DataFrame(sample_data)
        result_df = get_story(sample_df)

        self.assertEqual(result_df['story'].tolist(), [0, 0, 0])

    def test_get_story_multiple_category(self):
        text1 = 'The sun sets over the horizon, painting the sky with hues of orange and pink. Birds fly homeward, their silhouettes darkening against the vibrant backdrop. A sense of calm settles in as the world transitions from day to night.'
        text2 = 'As the day comes to a close, the sun dips below the edge of the earth, casting a warm glow across the landscape. Flocks of birds soar through the air, heading towards their nests. The tranquil beauty of twilight envelops everything in a serene embrace.'
        text3 = 'Beneath a sky painted with the last remnants of daylight, the sun bids its farewell, casting an amber glow over the surroundings. The air becomes cooler, and the sounds of rustling leaves and distant streams fill the atmosphere. Natures rhythm embraces the world in a soothing lullaby as night gradually takes hold.'
        sample_data = {'date': ['2019-10-01', '2019-10-10', '2019-10-03'],
                       'category': ['A', 'B', 'A'],
                       'text': [text1, text2, text3]
                        }
        sample_df = pd.DataFrame(sample_data)
        result_df = get_story(sample_df)

        self.assertEqual(result_df['story'].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
