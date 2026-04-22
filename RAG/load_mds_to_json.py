import os
import json

class PAPER_LOADER:
    __papers_dir: str = "./papers/"
    __data_path: str = "./data/papers.json"
    local_paper_contents: dict[str, str] # title: content

    def __init__(self):
        pass

    def load_local_paper_content(self) -> str:
        try:
            with open(self.__data_path, "r", encoding="utf-8") as file:
                self.local_paper_contents = json.load(file)
        except FileNotFoundError:
            self.local_paper_contents = {}

    def update_json(self) -> None:
        # 1. Load online paper contents 
        online_paper_contents: dict[str, str] = self.__get_online_paper_contents()
        # 2. Save updated local paper contents to file
        with open(self.__data_path, "w", encoding="utf-8") as file:
            json.dump(online_paper_contents, file, ensure_ascii=False, indent=2)

    def __get_online_paper_contents(self) -> dict[str, str]:
        # Extract paper titles
        paper_titles: list[str] = [filename for filename in os.listdir(self.__papers_dir) if os.path.isfile(os.path.join(self.__papers_dir, filename))]
        # Filter paper titles 
        paper_titles = [paper_title for paper_title in paper_titles if paper_title[:7] != "figures" and paper_title not in {"README.md", "To Read.md", "template.md"}]
        
        # Load paper contents
        online_paper_contents = {}
        for paper_title in paper_titles:
            content: str = self.__get_paper_content(paper_title)
            online_paper_contents[paper_title] = content

        return online_paper_contents

    def __get_paper_content(self, paper_title: str) -> str:
        print(os.path.join(self.__papers_dir, paper_title))
        with open(os.path.join(self.__papers_dir, paper_title), "r", encoding="utf-8") as file:
            content = file.read()
        return content
    


if __name__ == "__main__":
    paper_loader: PAPER_LOADER = PAPER_LOADER()
    # 1. Update local paper contents to JSON file
    paper_loader.update_json()
    # 2. Load local paper contents from JSON file
    paper_loader.load_local_paper_content()
    for title, content in paper_loader.local_paper_contents.items():
        print(f"{title}\n{'='*50}\n{content[:1000]}\n...")
        exit()
    

 