additional_name = additional_name.groupby('MUSEAAL_ID', as_index=True).agg({'TEKST':' '.join, 'LIIK':pd.Series.mode})
additional_name = additional_name.rename(columns={'TEKST':'text', 'LIIK':'class'})
description = description.groupby('MUSEAAL_ID', as_index=True).agg({'LISATEKST':' '.join})
description = description.rename(columns={'LISATEKST':'additional_text'})
# for now dont use all awailable measures
dimensions = dimensions.rename(columns={'PARAMEETER':'parameter','TAIS_NR':'full_nr', 'NIMETUS':'name', 'YHIK':'unit','VAARTUS':'value'}).drop(columns=['KOMMENTAAR'])
dimensions = dimensions.groupby('MUSEAAL_ID', as_index=True).first()[['parameter', 'unit', 'value']]
events = events.groupby('MUSEAAL_ID', as_index=True).first()
events = events.rename(columns={'TAIS_NR':'full_nr', 'SYNDMUSE_LIIK':'event_type', 'ASUKOHT':'location', 'ALGUS':'start', 'LOPP':'end', 'ENNE_KR':'before_Christ', 'RIIK_ADMIN_KOND':'country_and_unit', 'OSALEJA_ROLL':'participants_role', 'OSALEJA':'participant', 'KIHELKOND':'parish'}).drop(columns=['KUU'])
general_info = general_info.groupby('MUSEAAL_ID', as_index=True).first()
general_info = general_info.rename(columns={'ACR':'museum_abbr', 'TRT':'musealia_mark', 'TRS':'musealia_seria_nr', 'TRJ':'musealia_queue_nr', 'TRL':'musealia_additional_nr', 'KT':'collection_mark', 'KS':'ks', 'KJ':'collection_queue_nr','KL':'collection_additional_nr', 'NIMETUS':'name', 'KAHJUSTUSED':'damages', 'SEISUND':'state'}).drop(columns=['OLEMUS'])
materials = materials.rename(columns={'MATERJAL':'material', 'KOMMENTAAR':'commentary', 'TAIS_NR':'full_nr', 'NIMETUS':'name'})
materials['commentary'] = materials['commentary'].replace(np.nan,'')
materials = materials.groupby('MUSEAAL_ID', as_index=True).agg({'material':'>'.join, 'commentary':' '.join})
materials['commentary'] = materials['commentary'].replace('', np.nan)
technique = technique.rename(columns={'TEHNIKA':'technique','TAIS_NR':'full_nr', 'NIMETUS':'name'}).drop(columns=['KOMMENTAAR'])
technique = technique.groupby('MUSEAAL_ID', as_index=True).agg({'technique':'>'.join})
dfs = [additional_name, description, events, general_info, materials, technique, dimensions]

from functools import reduce
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['MUSEAAL_ID'],
                                            how='outer'), dfs)
df_merged.index.names=['id']