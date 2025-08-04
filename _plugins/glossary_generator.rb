require 'json'

Jekyll::Hooks.register :site, :post_write do |site|
  glossary_data = site.data['glossary']
  
  if glossary_data
    assets_data_dir = File.join(site.dest, 'assets', 'data')
    FileUtils.mkdir_p(assets_data_dir)
    
    glossary_json_path = File.join(assets_data_dir, 'glossary.json')
    File.write(glossary_json_path, JSON.pretty_generate(glossary_data))
    
    puts "Generated glossary.json with #{glossary_data.keys.length} terms"
  end
end
