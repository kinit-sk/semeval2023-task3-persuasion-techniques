"""
Class that encapsulates our Google Translation API. Currently it only translates to English, but different target languages can be configured.


# Usage

- DO translate in bulks. `GoogleTranslate.translate` saves the newly translated _input_data into csv. Try to send all the texts at the same time so that you minimize the time spent on writing to file.
- Use `enable_google_api=True` when you want to connect to the Google API. This is turned off by default to minimize spending by mistake.


By default, use it with `enable_api=False`. The model will return `None` for unknown texts:

```
translator = GoogleTranslate()
translator.translate(['Ahoj', 'Čo', 'Neznámy text'])
> [('Hi', 'sk'), ('What', 'sk'), None]
```

# Supported languages

As of 2022-07-18:
bh af ak sq am ar hy as ay az eu be bn bs bg ca ceb ny zh-CN zh-TW co hr cs da dv nl en eo et ee tl fi fr fy gl lg ka de el gn gu ht ha haw iw hi hmn hu is ig id ga it ja jw kn kk km rw ko kri ku ky lo la lv ln lt lb mk mg ms ml mt mi mr mn my ne nso no or om ps fa pl pt pa qu ro ru sm sa gd sr st sn sd si sk sl so es su sw sv tg ta tt te th ti ts tr tk uk ur ug uz vi cy xh yi yo zu he zh

The full current list can be obtained via:

1. `GoogleTranslate(enable_api=True).client.get_languages()`. This call actually sends request to API so it should not be abused.
2. https://cloud.google.com/translate/docs/languages


# Python API documentation

https://googleapis.dev/python/translation/latest/client.html

"""
import logging
import os
from typing import Dict, List, Union

# v2 is a basic translation, there is also v3, but it's not needed for our use-cases
# See: https://cloud.google.com/translate/docs/editions
from google.cloud import translate_v2
import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


class GoogleTranslate:
    
    KEY_PATH = os.path.join('config', 'placeholder') #
    TRANSLATIONS_PATH = os.path.join('_input_data', 'translations.csv')
    
    
    def __init__(self, enable_api=False):
        self.logger = logging.getLogger(__name__)
        self.loaded = False
        self.enable_api = enable_api
        
        if self.enable_api:
            self.client = translate_v2.Client.from_service_account_json(self.KEY_PATH)

        # TODO: Check remote version of the `translations.csv`?
        
    
    def load(self):
        self.dataframe = pd.read_csv(self.TRANSLATIONS_PATH, na_values=[]).set_index('original_text')
        self.dict = self.load_dict()
        self.loaded = True
        return self
    
    
    def save(self):
        self.dataframe.to_csv(self.TRANSLATIONS_PATH)
        
        
    def load_dict(self):
        return dict(
            (original_text, (english_text, detected_language))
            for original_text, english_text, detected_language in zip(self.dataframe.index, self.dataframe['english_text'], self.dataframe['detected_language'])
        )

    def _api_translate_text(self, text: str) -> Dict:
        """
        Translate one document and return a well-formed Dict
        """
        
        if not self.enable_api:
            raise RuntimeError(f'This translator object does not have an API access enabled. Use `GoogleTranslate(enable_api=True)` if you wish to access Google API. You tried to translate: "{text}"')
        
        response = self.client.translate(
            text,
            target_language='en',
            format_='text',  # `text` is needed because default `format_='html'` escapes special characters
        )
        
        return {
            'original_text': response['input'],
            'english_text': response['translatedText'],
            'detected_language': response['detectedSourceLanguage'],
        }
    
    
    def translate(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Translate str or List[str] to English. First check whether the translation already happened and is the `self.dataframe`. All untranslated texts are sent to the API and saved to the local file. List of translated texts is returned, but we also save detected languages.
        """
        is_list = True
        if isinstance(texts, str):
            texts = [texts]
            is_list = False
        
        if not self.loaded:
            self.load()
                    
        untranslated = [t for t in set(texts) if t not in self.dict]
        
        if untranslated and self.enable_api:
            self.logger.warning(f'{len(untranslated)}/{len(set(texts))} translations missing')
            i = 0
            while untranslated[i*1000:(i+1)*1000]:
                self.logger.warning(f'Translating batch no {i+1}')
                self._api_translate_batch(untranslated[i*1000:(i+1)*1000])
                i += 1
            untranslated = list(set(texts) - set(self.dict.keys()))
        
        if untranslated:
            self.logger.warning(f'{len(untranslated)}/{len(set(texts))} translations missing')
        
        if is_list:
            return [
                self.dict.get(text, None)
                for text in texts
            ]
        else:
            return self.dict.get(texts[0], None)

        
    def _api_translate_batch(self, texts: List[str]) -> None:
        """
        Create the translation dataframe for new texts. If any error happens with a request, the script continutes with remaining texts. There are some cases where this happens, e.g. texts in Sorani Kurdish cause BadRequest (as of July 2022). `self.dataframe`, `self.dict` and `tranlations.csv` file are all updated.
        
        This function could be optimized in the future by sending multiple texts at the same time. According to API documentation, 5K character requests are optimal: https://cloud.google.com/translate/quotas
        """
        
        translations = []
        for text in tqdm(texts):
            try:
                translations.append(self._api_translate_text(text))
            except Exception as e:
                self.logger.error(f'\n\n\nERROR TEXT: {text}\n\n\nERROR MESSAGE: {e}')
        
        # If at least one translation was successful, update dataframe and save to file.
        if translations:
            self.dataframe = pd.concat([
                self.dataframe,
                pd.DataFrame(translations).set_index('original_text')
            ])
            self.save()
            self.dict = self.load_dict()


"""
Credits are due to Matus Pikuliak for creating this class
"""