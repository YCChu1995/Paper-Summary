import os

class PAPER_LOADER:
    __papers_dir: str = "./papers/"
    __data_path: str = "./data/papers.csv"
    __online_paper_contents: dict[str, str] # title: content
    local_paper_contents: dict[str, str] # title: content

    def __init__(self):
        self.__load_local_paper_content()

    def __load_local_paper_content(self) -> str:
        with open(self.__data_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if len(lines) <= 1:
                self.local_paper_contents = {}
            else:
                self.local_paper_contents = {line.split(",")[0]: line.split(",")[1].strip() for line in lines[1:]} # Skip the header line

    def update_local_paper_contents(self) -> None:
        # 1. Load online paper contents 
        self.__get_online_paper_contents()
        # 2. Save updated local paper contents to file
        with open(self.__data_path, "w", encoding="utf-8") as file:
            file.write("title,content\n") # Write header line
            for title, content in self.__online_paper_contents.items():
                file.write(f"{title},{content}\n")

    def __get_online_paper_contents(self):
        # Extract paper titles
        paper_titles: list[str] = [filename for filename in os.listdir(self.__papers_dir) if os.path.isfile(os.path.join(self.__papers_dir, filename))]
        # Filter paper titles 
        paper_titles = [paper_title for paper_title in paper_titles if paper_title[:7] != "figures" and paper_title not in {"README.md", "To Read.md", "template.md"}]
        
        # Load paper contents
        self.__online_paper_contents = {}
        for paper_title in paper_titles:
            content: str = self.__get_paper_content(paper_title)
            self.__online_paper_contents[paper_title] = content

    def __get_paper_content(self, paper_title: str) -> str:
        print(os.path.join(self.__papers_dir, paper_title))
        with open(os.path.join(self.__papers_dir, paper_title), "r", encoding="utf-8") as file:
            content = file.read()
        return content
    


if __name__ == "__main__":
    paper_loader: PAPER_LOADER = PAPER_LOADER()
    paper_loader.update_local_paper_contents()
    for title, content in paper_loader.local_paper_contents.items():
        print(f"Title: {title}, Content: {content[:100]}")

    # # 1. Load online paper contents
    # paper_loader.get_online_paper_contents()
    # for title, content in paper_loader.online_paper_contents.items():
    #     print(f"Title: {title}, Content: {content[:100]}")
    

 