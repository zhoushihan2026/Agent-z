import asyncio
from browser_use import Browser, BrowserConfig


async def main():
    browser = Browser(config=BrowserConfig(headless=False))
    try:
        context = await browser.new_context()
        await context.navigate_to('https://example.com')
        state = await context.get_state()
        print('title=', state.title if state else '')
        print('url=', state.url if state else '')
    finally:
        await browser.close()


asyncio.run(main())
