from utils.donut_utils import load_donut_model_and_processor, prepare_data_using_processor
import pkg_resources
from symspellpy import SymSpell
from word2number import w2n
from dateutil import relativedelta
from datetime import datetime
from word2number import w2n
import re

CHEQUE_PARSER_MODEL = "shivi/donut-cheque-parser"
TASK_PROMPT = "<parse-cheque>"

def parse_cheque_with_donut(input_image_path):

    donut_processor, model = load_donut_model_and_processor(CHEQUE_PARSER_MODEL)

    cheque_image_tensor, input_for_decoder = prepare_data_using_processor(donut_processor,input_image_path,TASK_PROMPT)
    
    outputs = model.generate(cheque_image_tensor,
                                decoder_input_ids=input_for_decoder,
                                max_length=model.decoder.config.max_position_embeddings,
                                early_stopping=True,
                                pad_token_id=donut_processor.tokenizer.pad_token_id,
                                eos_token_id=donut_processor.tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1,
                                bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True,
                                output_scores=True,)

    decoded_output_sequence = donut_processor.batch_decode(outputs.sequences)[0]
    
    extracted_cheque_details = decoded_output_sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
    ## remove task prompt from token sequence
    cleaned_cheque_details = re.sub(r"<.*?>", "", extracted_cheque_details, count=1).strip()  
    ## generate ordered json sequence from output token sequence
    cheque_details_json = donut_processor.token2json(cleaned_cheque_details)
    print("cheque_details_json:",cheque_details_json['cheque_details'])
    
    ## extract required fields from predicted json

    amt_in_words  = cheque_details_json['cheque_details'][0]['amt_in_words']
    amt_in_figures = cheque_details_json['cheque_details'][1]['amt_in_figures']
    macthing_amts = match_legal_and_courstesy_amount(amt_in_words,amt_in_figures)
    
    payee_name = cheque_details_json['cheque_details'][2]['payee_name']

    bank_name = cheque_details_json['cheque_details'][3]['bank_name']
    cheque_date = cheque_details_json['cheque_details'][4]['cheque_date']

    stale_cheque = check_if_cheque_is_stale(cheque_date)

    return payee_name,amt_in_words,amt_in_figures,bank_name,cheque_date,macthing_amts,stale_cheque

def spell_check(amt_in_words):
    sym_spell = SymSpell(max_dictionary_edit_distance=2,prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    suggestions = sym_spell.lookup_compound(amt_in_words, max_edit_distance=2)

    return suggestions[0].term

def match_legal_and_courstesy_amount(legal_amount,courtesy_amount):
    macthing_amts = False
    if len(legal_amount) == 0:
        return macthing_amts

    corrected_amt_in_words = spell_check(legal_amount)
    print("corrected_amt_in_words:",corrected_amt_in_words)
    
    numeric_legal_amt = w2n.word_to_num(corrected_amt_in_words)
    print("numeric_legal_amt:",numeric_legal_amt)
    if int(numeric_legal_amt) == int(courtesy_amount):
        macthing_amts = True
    return macthing_amts

def check_if_cheque_is_stale(cheque_issue_date):
    stale_check = False
    current_date = datetime.now().strftime('%d/%m/%y')
    current_date_ = datetime.strptime(current_date, "%d/%m/%y")
    cheque_issue_date_ = datetime.strptime(cheque_issue_date, "%d/%m/%y")
    relative_diff = relativedelta.relativedelta(current_date_, cheque_issue_date_)
    months_difference = (relative_diff.years * 12) + relative_diff.months
    print("months_difference:",months_difference)
    if months_difference > 3:
        stale_check = True
    return stale_check